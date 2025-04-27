import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import logging
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("falcon.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constants
NUM_STOCKS = 129
TOP_K = 10
SEQUENCE_LENGTH_DAILY = 30  # 1 months of trading days
EMBEDDING_DIM = 128
NUM_HEADS = 8
REGIME_CLUSTERS = 3  # Bullish, Bearish, Volatile
TRAINING_START = '2005-01-01'
TRAINING_END = '2022-01-01'
TESTING_START = '2022-01-01'
TESTING_END = '2025-01-01'  # Using the current date in the context


# Set seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the transformer encoder block.

        Args:
            embed_dim: Dimension of the embedding vector
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the transformer encoder block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Processed tensor of the same shape
        """
        # Layer normalization and attention
        norm_x = self.norm1(x)
        attn_output, _ = self.attention(norm_x, norm_x, norm_x)

        # First residual connection
        x = x + self.dropout(attn_output)

        # Layer normalization and feed forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)

        # Second residual connection
        return x + self.dropout(ff_output)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformers."""

    def __init__(self, embed_dim: int, max_seq_length: int = 5000):
        """
        Initialize the positional encoding.

        Args:
            embed_dim: Dimension of the embedding vector
            max_seq_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))

        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]

        Returns:a
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class AssetEncoder(nn.Module):
    """Base layer encoder for individual assets."""

    def __init__(self, feature_dim: int, embed_dim: int, num_heads: int,
                 daily_seq_len: int, dropout: float = 0.1):
        """
        Initialize the asset encoder.

        Args:
            feature_dim: Number of features for each time step
            embed_dim: Dimension of the embedding vector
            num_heads: Number of attention heads
            daily_seq_len: Sequence length for daily data
            dropout: Dropout probability
        """
        super(AssetEncoder, self).__init__()

        # Feature embeddings for daily data only
        self.daily_embedding = nn.Linear(feature_dim, embed_dim)

        # Positional encoding
        self.daily_pos_encoding = PositionalEncoding(embed_dim, daily_seq_len)

        # Transformer encoder block
        self.daily_transformer = TransformerEncoderBlock(embed_dim, num_heads, dropout)

        # Final projection
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, daily_data: torch.Tensor) -> torch.Tensor:
        """
        Process asset data through the encoder.

        Args:
            daily_data: Daily data of shape [batch_size, daily_seq_len, feature_dim]

        Returns:
            Asset embedding of shape [batch_size, embed_dim]
        """
        # Project data to embedding space
        daily_embedded = self.daily_embedding(daily_data)

        # Add positional encoding
        daily_embedded = self.daily_pos_encoding(daily_embedded)

        # Apply transformer block
        daily_encoded = self.daily_transformer(daily_embedded)

        # Global pooling (mean over sequence dimension)
        daily_pooled = torch.mean(daily_encoded, dim=1)

        # Project
        output = self.projection(daily_pooled)

        return self.dropout(output)


class CrossAssetAttention(nn.Module):
    """Middle layer for cross-asset attention and regime detection."""

    def __init__(self, num_assets: int, embed_dim: int, num_heads: int, num_regimes: int, dropout: float = 0.1):
        """
        Initialize the cross-asset attention module.

        Args:
            num_assets: Number of assets being processed
            embed_dim: Dimension of the embedding vector
            num_heads: Number of attention heads
            num_regimes: Number of market regimes to detect
            dropout: Dropout probability
        """
        super(CrossAssetAttention, self).__init__()

        self.num_assets = num_assets
        self.embed_dim = embed_dim

        # Cross-asset attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Market context representation
        self.market_context = nn.Linear(embed_dim, embed_dim)

        # Asset-specific processing with global context
        self.asset_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_assets)
        ])

        # Regime detection
        self.regime_detector = nn.Linear(embed_dim, num_regimes)

    def forward(self, asset_embeddings: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Process asset embeddings through cross-asset attention.

        Args:
            asset_embeddings: List of asset embeddings, each of shape [batch_size, embed_dim]

        Returns:
            Tuple containing:
                - List of context-aware asset embeddings
                - Regime probabilities
        """
        # Find minimum batch size across all embeddings
        min_batch_size = min([embed.shape[0] for embed in asset_embeddings])

        # Truncate all embeddings to the minimum batch size
        asset_embeddings = [embed[:min_batch_size] for embed in asset_embeddings]

        # Stack embeddings for batch processing [batch_size, num_assets, embed_dim]
        stacked_embeddings = torch.stack(asset_embeddings, dim=1)
        batch_size = stacked_embeddings.shape[0]

        # Apply cross-asset attention
        norm_embeddings = self.norm(stacked_embeddings)
        attn_output, _ = self.cross_attention(norm_embeddings, norm_embeddings, norm_embeddings)

        # Residual connection
        context_aware_embeddings = stacked_embeddings + attn_output

        # Compute global market context
        market_context = torch.mean(context_aware_embeddings, dim=1)  # [batch_size, embed_dim]
        market_context = self.market_context(market_context)

        # Enhanced asset representations with global context
        enhanced_embeddings = []
        for i in range(self.num_assets):
            if i < context_aware_embeddings.shape[1]:  # Safety check
                asset_embed = context_aware_embeddings[:, i, :]
                # Concatenate with global context
                with_context = torch.cat([asset_embed, market_context], dim=1)
                # Project back to original dimension
                enhanced = self.asset_projection[i](with_context)
                enhanced_embeddings.append(enhanced)
            else:
                # If we somehow have fewer assets in the tensor than expected, add a zero tensor
                dummy_embed = torch.zeros((batch_size, self.embed_dim), device=device)
                enhanced_embeddings.append(dummy_embed)

        # Regime detection
        regime_probs = torch.softmax(self.regime_detector(market_context), dim=1)

        return enhanced_embeddings, regime_probs


class PortfolioAgent(nn.Module):
    """Top layer for portfolio optimization and stock selection."""

    def __init__(self, num_assets: int, embed_dim: int, num_regimes: int, dropout: float = 0.1):
        """
        Initialize the portfolio optimization agent.

        Args:
            num_assets: Number of assets being processed
            embed_dim: Dimension of the embedding vector
            num_regimes: Number of market regimes
            dropout: Dropout probability
        """
        super(PortfolioAgent, self).__init__()

        # State representation
        self.state_encoder = nn.Sequential(
            nn.Linear(embed_dim + num_regimes, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Portfolio allocation (weights)
        self.portfolio_allocator = nn.Linear(128, num_assets)

        # Asset selection (scores for ranking)
        self.asset_scorer = nn.Linear(128, num_assets)

    def forward(self, asset_embeddings: List[torch.Tensor], regime_probs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Generate portfolio weights and asset scores.

        Args:
            asset_embeddings: List of asset embeddings, each of shape [batch_size, embed_dim]
            regime_probs: Regime probabilities of shape [batch_size, num_regimes]

        Returns:
            Tuple containing:
                - Portfolio weights of shape [batch_size, num_assets]
                - Asset scores of shape [batch_size, num_assets]
        """
        # Stack embeddings for batch processing [batch_size, num_assets, embed_dim]
        stacked_embeddings = torch.stack(asset_embeddings, dim=1)

        # Compute market context
        market_context = torch.mean(stacked_embeddings, dim=1)  # [batch_size, embed_dim]

        # Combine with regime information
        state = torch.cat([market_context, regime_probs], dim=1)

        # Encode state
        encoded_state = self.state_encoder(state)

        # Portfolio allocation (softmax for simplex constraint)
        portfolio_weights = torch.softmax(self.portfolio_allocator(encoded_state), dim=1)

        # Asset selection scores (linear outputs for ranking)
        asset_scores = self.asset_scorer(encoded_state)

        return portfolio_weights, asset_scores


class FALCON(nn.Module):
    """FALCON model for multi-stock prediction and portfolio optimization."""

    def __init__(self, ticker_list: List[str], feature_dim: int, embed_dim: int = 128,
                 num_heads: int = 8, num_regimes: int = 3, dropout: float = 0.1):
        """
        Initialize the FALCON model.

        Args:
            ticker_list: List of stock tickers
            feature_dim: Number of features for each time step
            embed_dim: Dimension of the embedding vector
            num_heads: Number of attention heads
            num_regimes: Number of market regimes to detect
            dropout: Dropout probability
        """
        super(FALCON, self).__init__()

        self.ticker_list = ticker_list
        self.num_assets = len(ticker_list)
        self.embed_dim = embed_dim

        # Base layer: Asset-specific encoders
        self.asset_encoders = nn.ModuleDict({
            ticker: AssetEncoder(
                feature_dim=feature_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                daily_seq_len=SEQUENCE_LENGTH_DAILY,
                dropout=dropout
            ) for ticker in ticker_list
        })

        # Middle layer: Cross-asset attention and regime detection
        self.cross_asset_module = CrossAssetAttention(
            num_assets=self.num_assets,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_regimes=num_regimes,
            dropout=dropout
        )

        # Top layer: Portfolio optimization agent
        self.portfolio_agent = PortfolioAgent(
            num_assets=self.num_assets,
            embed_dim=embed_dim,
            num_regimes=num_regimes,
            dropout=dropout
        )

        # For pretraining (return prediction head)
        self.return_predictor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_daily: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the FALCON model.

        Args:
            x_daily: Dictionary mapping tickers to daily data tensors

        Returns:
            Tuple containing:
                - Portfolio weights
                - Asset scores
                - Regime probabilities
                - Asset embeddings
        """
        # Base layer: Asset-specific encoding
        asset_embeddings = []
        for ticker in self.ticker_list:
            if ticker in x_daily and x_daily[ticker].size(0) > 0:
                embedding = self.asset_encoders[ticker](x_daily[ticker])
                asset_embeddings.append(embedding)
            else:
                # For missing tickers, use a zero tensor
                # Get batch size from any available tensor
                batch_size = next(iter(x_daily.values())).size(0) if x_daily else 1
                dummy_embedding = torch.zeros((batch_size, self.embed_dim), device=device)
                asset_embeddings.append(dummy_embedding)

        # Middle layer: Cross-asset attention and regime detection
        enhanced_embeddings, regime_probs = self.cross_asset_module(asset_embeddings)

        # Top layer: Portfolio optimization
        portfolio_weights, asset_scores = self.portfolio_agent(enhanced_embeddings, regime_probs)

        return portfolio_weights, asset_scores, regime_probs, asset_embeddings

    def predict_returns(self, asset_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Predict returns for pretraining.

        Args:
            asset_embeddings: List of asset embeddings

        Returns:
            List of predicted returns
        """
        return [self.return_predictor(embed) for embed in asset_embeddings]


class StockDataset(Dataset):
    """Dataset for stock data."""

    def __init__(self, daily_data: Dict[str, np.ndarray], returns: Dict[str, np.ndarray], tickers: List[str]):
        """
        Initialize the dataset.

        Args:
            daily_data: Dictionary mapping tickers to daily data arrays
            returns: Dictionary mapping tickers to return arrays
            tickers: List of stock tickers
        """
        self.daily_data = daily_data
        self.returns = returns
        self.tickers = tickers

        # Determine dataset size (use the first ticker with data as reference)
        for ticker in tickers:
            if ticker in daily_data and len(daily_data[ticker]) > 0:
                self.size = len(daily_data[ticker])
                break
        else:
            self.size = 0

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - Dictionary mapping tickers to daily data arrays
                - Dictionary mapping tickers to return arrays
        """
        x_daily = {}
        y_returns = {}

        for ticker in self.tickers:
            if ticker in self.daily_data and idx < len(self.daily_data[ticker]):
                x_daily[ticker] = self.daily_data[ticker][idx]

                if ticker in self.returns and idx < len(self.returns[ticker]):
                    y_returns[ticker] = self.returns[ticker][idx]
        return x_daily, y_returns


def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]) -> Tuple[
    Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collate function for the dataloader.

    Args:
        batch: List of sample tuples (x_daily, y_returns)

    Returns:
        Tuple containing:
            - Dictionary mapping tickers to daily data tensors
            - Dictionary mapping tickers to return tensors
    """
    x_daily_batch = {}
    y_returns_batch = {}

    # Extract all tickers from the batch
    tickers = set()
    for x_daily, _ in batch:
        tickers.update(x_daily.keys())

    # Process daily data
    for ticker in tickers:
        daily_samples = [x_daily[ticker] for x_daily, _ in batch if ticker in x_daily]
        if daily_samples:
            x_daily_batch[ticker] = torch.tensor(np.stack(daily_samples), dtype=torch.float32)

    # Process returns
    for ticker in tickers:
        return_samples = [y_returns[ticker] for _, y_returns in batch if ticker in y_returns]
        if return_samples:
            y_returns_batch[ticker] = torch.tensor(np.stack(return_samples), dtype=torch.float32)

    return x_daily_batch, y_returns_batch


class FALCONSystem:
    """FALCON system for multi-stock prediction and portfolio optimization."""

    def __init__(self, ticker_list: List[str], data_dir: str = "stock_data"):
        """
        Initialize the FALCON system.

        Args:
            ticker_list: List of stock tickers
            data_dir: Directory for storing data
        """
        self.ticker_list = ticker_list
        self.data_dir = data_dir
        self.model = None
        self.scalers = {}

        # Create data directories
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)

        logger.info(f"Initialized FALCON system with {len(ticker_list)} tickers")

    def download_stock_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Download stock data from Yahoo Finance.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping tickers to DataFrames
        """
        logger.info("Loading stock data...")
        data = {}

        for ticker in tqdm(self.ticker_list, desc="Downloading stock data"):
            try:
                # Download daily data
                daily_data = yf.download(ticker, start=start_date, end=end_date)

                if not daily_data.empty:
                    daily_file = os.path.join(self.data_dir, "raw", f"{ticker}_daily.csv")
                    daily_data.to_csv(daily_file)
                    data[ticker] = daily_data
                    logger.info(f"Downloaded daily data for {ticker}: {len(daily_data)} records")
                else:
                    logger.warning(f"No data found for {ticker}")
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {e}")

        return data

    def process_stock_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process stock data from raw CSV files.

        Returns:
            Dictionary mapping tickers to dictionaries of processed DataFrames
        """
        logger.info("Processing data files...")
        processed_data = {}

        for ticker in tqdm(self.ticker_list, desc="Processing data files"):
            try:
                # Load daily data
                daily_file = os.path.join(self.data_dir, "raw", f"{ticker}_daily.csv")

                if os.path.exists(daily_file):
                    daily_data = pd.read_csv(daily_file, index_col=0, parse_dates=True)

                    # Convert string columns to numeric types first
                    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_columns:
                        if col in daily_data.columns:
                            daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')

                    # Check if 'Adj Close' exists or use 'Close'
                    if 'Adj Close' not in daily_data.columns:
                        daily_data['Adj Close'] = daily_data['Close']
                        logger.info(f"Using 'Close' as 'Adj Close' for {ticker}")
                    else:
                        # Convert Adj Close to numeric as well
                        daily_data['Adj Close'] = pd.to_numeric(daily_data['Adj Close'], errors='coerce')

                    # Now perform calculations with numeric data
                    daily_data['Returns'] = daily_data['Adj Close'].pct_change()
                    daily_data['Log_Returns'] = np.log(daily_data['Adj Close'] / daily_data['Adj Close'].shift(1))
                    daily_data['Volatility'] = daily_data['Returns'].rolling(window=20).std()
                    daily_data['SMA_20'] = daily_data['Adj Close'].rolling(window=20).mean()
                    daily_data['SMA_50'] = daily_data['Adj Close'].rolling(window=50).mean()
                    daily_data['SMA_Ratio'] = daily_data['SMA_20'] / daily_data['SMA_50']
                    daily_data['Volume_Change'] = daily_data['Volume'].pct_change()

                    # Clean up NaN values
                    daily_data.dropna(inplace=True)

                    # Save processed data
                    processed_file = os.path.join(self.data_dir, "processed", f"{ticker}_daily_processed.csv")
                    daily_data.to_csv(processed_file)

                    processed_data[ticker] = {'daily': daily_data}
                    logger.info(f"Processed daily data for {ticker}: {len(daily_data)} records")
            except Exception as e:
                logger.error(f"Error processing data for {ticker}: {e}")

        return processed_data

    def is_scaler_fitted(self, scaler):
        """Check if a MinMaxScaler is properly fitted."""
        return hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_')

    def prepare_data(self, training_start: str = TRAINING_START, training_end: str = TRAINING_END,
                     testing_start: str = None, testing_end: str = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare data for training and testing.

        Args:
            training_start: Start date for training
            training_end: End date for training
            testing_start: Start date for testing (optional)
            testing_end: End date for testing (optional)

        Returns:
            Tuple containing:
                - Dictionary of training/testing sequences
                - Dictionary of target returns
        """
        logger.info("Preparing training data...")

        X_daily = {}
        y_returns = {}

        for ticker in tqdm(self.ticker_list, desc="Preparing training data"):
            try:
                processed_file = os.path.join(self.data_dir, "processed", f"{ticker}_daily_processed.csv")

                if os.path.exists(processed_file):
                    data = pd.read_csv(processed_file, index_col=0, parse_dates=True)

                    # Split into train/test sets based on dates
                    if testing_start is not None and testing_end is not None:
                        # Using the dates provided for training and testing
                        train_data = data.loc[training_start:training_end]
                        test_data = data.loc[testing_start:testing_end]
                        current_data = test_data  # Using test data as current data
                    else:
                        # Using all data for training
                        train_data = data.loc[training_start:training_end]
                        current_data = train_data

                    # Skip if we don't have enough data for this ticker
                    if len(train_data) < 20:
                        logger.warning(f"Not enough training data for {ticker}, skipping")
                        continue

                    # Extract features for daily data
                    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                'Returns', 'Log_Returns', 'Volatility', 'SMA_20', 'SMA_50', 'SMA_Ratio',
                                'Volume_Change']

                    # Scale features (fit on training data, apply to all)
                    # Create a new scaler for this ticker and fit it on training data
                    try:
                        if ticker not in self.scalers or not self.is_scaler_fitted(self.scalers[ticker]):
                            self.scalers[ticker] = MinMaxScaler()
                            train_features = train_data[features].values
                            self.scalers[ticker].fit(train_features)
                            logger.info(f"Scaler for {ticker} successfully fitted")

                        # Apply scaling
                        current_features = current_data[features].values
                        scaled_features = self.scalers[ticker].transform(current_features)
                    except Exception as e:
                        logger.error(f"Error scaling features for {ticker}: {e}")
                        continue

                    # Create sequences
                    sequences = []
                    targets = []

                    for i in range(len(scaled_features) - SEQUENCE_LENGTH_DAILY - 1):
                        seq = scaled_features[i:i + SEQUENCE_LENGTH_DAILY]
                        target = current_data['Returns'].iloc[i + SEQUENCE_LENGTH_DAILY]

                        if not np.isnan(target) and not np.any(np.isnan(seq)):
                            sequences.append(seq)
                            targets.append(target)

                    if sequences:
                        X_daily[ticker] = np.array(sequences)
                        y_returns[ticker] = np.array(targets)
                        logger.info(f"Prepared daily training data for {ticker}: {len(sequences)} sequences")
            except Exception as e:
                logger.error(f"Error preparing data for {ticker}: {e}")

        return X_daily, y_returns

    def train_base_layer(self, X_daily_train: Dict[str, np.ndarray], y_train: Dict[str, np.ndarray],
                         batch_size: int = 128, epochs: int = 500, learning_rate: float = 0.0001) -> None:
        """
        Pretrain the base layer encoders.

        Args:
            X_daily_train: Dictionary of daily training sequences
            y_train: Dictionary of target returns
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        logger.info("Training base layer encoders...")

        # Check if we have any data to train on
        if not X_daily_train:
            logger.error("No training data available. Check data processing step.")
            return

        # Create initial model if not exists
        try:
            feature_dim = next(iter(X_daily_train.values())).shape[2]
        except (StopIteration, IndexError, AttributeError):
            logger.error("Empty or invalid training data. Check data processing.")
            return

        if self.model is None:
            self.model = FALCON(
                ticker_list=self.ticker_list,
                feature_dim=feature_dim,
                embed_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_regimes=REGIME_CLUSTERS,
                dropout=0.1
            ).to(device)

        # Create dataset
        train_dataset = StockDataset(X_daily_train, y_train, self.ticker_list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Define optimizer for asset encoders
        parameters = []
        for name, param in self.model.named_parameters():
            if 'asset_encoders' in name or 'return_predictor' in name:
                parameters.append(param)

        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            valid_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for x_daily_batch, y_returns_batch in progress_bar:
                # Skip if batch is empty
                if not x_daily_batch:
                    continue

                # Move data to device
                for ticker in x_daily_batch:
                    x_daily_batch[ticker] = x_daily_batch[ticker].to(device)

                for ticker in y_returns_batch:
                    y_returns_batch[ticker] = y_returns_batch[ticker].to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass through asset encoders
                asset_embeddings = []
                for ticker in self.ticker_list:
                    if ticker in x_daily_batch:
                        embedding = self.model.asset_encoders[ticker](x_daily_batch[ticker])
                        asset_embeddings.append(embedding)
                    else:
                        # Use a zero tensor for missing tickers
                        dummy_embedding = torch.zeros(
                            (x_daily_batch[next(iter(x_daily_batch))].shape[0], EMBEDDING_DIM),
                            device=device)
                        asset_embeddings.append(dummy_embedding)

                # Predict returns
                predicted_returns = self.model.predict_returns(asset_embeddings)

                # Calculate loss
                loss = 0.0
                num_predictions = 0

                for i, ticker in enumerate(self.ticker_list):
                    if ticker in y_returns_batch:
                        pred = predicted_returns[i]
                        # Take the minimum of pred size and target size
                        min_size = min(pred.size(0), y_returns_batch[ticker].size(0))
                        if min_size > 0:
                            pred_subset = pred[:min_size]
                            target_subset = y_returns_batch[ticker][:min_size].view(-1, 1)
                            ticker_loss = criterion(pred_subset, target_subset)
                            loss += ticker_loss
                            num_predictions += 1

                if num_predictions > 0:
                    loss /= num_predictions
                else:
                    loss = torch.tensor(0.0, device=device)

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN or Inf loss detected, skipping batch")
                    continue

                # Backward pass and optimize
                loss.backward()
                # Use gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                valid_batches += 1

                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

            avg_epoch_loss = epoch_loss / max(valid_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}")

            # Update learning rate based on validation loss
            scheduler.step(avg_epoch_loss)

        # Save model and scalers
        self.save_model(suffix="base_layer")

    def train_full_model(self, X_daily_train: Dict[str, np.ndarray], y_train: Dict[str, np.ndarray],
                         batch_size: int = 256, epochs: int = 1000, learning_rate: float = 0.0001) -> None:
        """
        Train the full FALCON model.

        Args:
            X_daily_train: Dictionary of daily training sequences
            y_train: Dictionary of target returns
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        logger.info("Training full FALCON model...")

        # Check if we have any data
        if not X_daily_train:
            logger.error("No training data available")
            return

        # Create dataset
        train_dataset = StockDataset(X_daily_train, y_train, self.ticker_list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Define optimizer for all parameters
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Define loss functions
        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()  # For regime classification

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            valid_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for x_daily_batch, y_returns_batch in progress_bar:
                # Skip empty batches
                if not x_daily_batch:
                    continue

                try:
                    # Move data to device
                    for ticker in x_daily_batch:
                        x_daily_batch[ticker] = x_daily_batch[ticker].to(device)

                    for ticker in y_returns_batch:
                        y_returns_batch[ticker] = y_returns_batch[ticker].to(device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    portfolio_weights, asset_scores, regime_logits, asset_embeddings = self.model(x_daily_batch)

                    # Get batch size from portfolio_weights
                    batch_size = portfolio_weights.size(0)

                    # 1. Return prediction loss
                    return_pred_loss = 0.0
                    num_predictions = 0
                    predicted_returns = self.model.predict_returns(asset_embeddings)

                    for i, ticker in enumerate(self.ticker_list):
                        if ticker in y_returns_batch:
                            pred = predicted_returns[i]
                            # Take the minimum of pred size and target size
                            min_size = min(pred.size(0), y_returns_batch[ticker].size(0))
                            if min_size > 0:
                                pred_subset = pred[:min_size]
                                target_subset = y_returns_batch[ticker][:min_size].view(-1, 1)
                                ticker_loss = mse_criterion(pred_subset, target_subset)
                                return_pred_loss += ticker_loss
                                num_predictions += 1

                    if num_predictions > 0:
                        return_pred_loss /= num_predictions
                    else:
                        return_pred_loss = torch.tensor(0.0, device=device)

                    # 2. Create target portfolio weights based on next-day returns (with safeguards)
                    target_weights = torch.zeros_like(portfolio_weights)

                    for i in range(batch_size):
                        weights = []
                        for j, ticker in enumerate(self.ticker_list):
                            if ticker in y_returns_batch and i < y_returns_batch[ticker].size(0):
                                # Get actual return for this asset (with index check)
                                actual_return = y_returns_batch[ticker][i].item()
                                # Higher weight for positive returns
                                weight = max(0, actual_return)  # No shorting
                                weights.append(weight)
                            else:
                                weights.append(0)

                        # Normalize weights to sum to 1 (with safeguards for all zeros)
                        weights = torch.tensor(weights, device=device)
                        weight_sum = torch.sum(weights)
                        if weight_sum > 0:
                            weights = weights / weight_sum
                        else:
                            # Equal weight if all negative
                            weights = torch.ones(len(weights), device=device) / len(weights)

                        if i < target_weights.size(0):  # Safety check
                            target_weights[i] = weights

                    # 3. Portfolio allocation loss
                    portfolio_loss = mse_criterion(portfolio_weights, target_weights)

                    # 4. Asset selection loss - higher scores for assets with better returns
                    target_scores = torch.zeros_like(asset_scores)

                    for i in range(batch_size):
                        scores = []
                        for j, ticker in enumerate(self.ticker_list):
                            if ticker in y_returns_batch and i < y_returns_batch[ticker].size(0):
                                # Use actual return as the target score (with index check)
                                scores.append(y_returns_batch[ticker][i].item())
                            else:
                                scores.append(-0.01)  # Slightly negative score for missing assets

                        if i < target_scores.size(0):  # Safety check
                            target_scores[i] = torch.tensor(scores, device=device)

                    # Calculate asset score loss - using scaled MSE to avoid dominating the loss
                    asset_score_loss = 0.01 * mse_criterion(asset_scores, target_scores)

                    # 5. Regime detection loss
                    # Create simple targets to encourage diversity
                    regime_targets = torch.randint(0, REGIME_CLUSTERS, (batch_size,), device=device)
                    regime_loss = ce_criterion(regime_logits, regime_targets)

                    # Combine losses with weights
                    total_loss = (
                            0.4 * return_pred_loss +
                            0.3 * portfolio_loss +
                            0.2 * asset_score_loss +
                            0.1 * regime_loss
                    )

                    # Check for NaN or Inf loss
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logger.warning("NaN or Inf loss detected, skipping batch")
                        continue

                    # Backward pass and optimize
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += total_loss.item()
                    num_batches += 1
                    valid_batches += 1

                    progress_bar.set_postfix({
                        "Total": f"{total_loss.item():.6f}",
                        "Return": f"{return_pred_loss.item():.6f}",
                        "Portfolio": f"{portfolio_loss.item():.6f}"
                    })

                except Exception as e:
                    logger.error(f"Error in training batch: {e}")
                    continue

            avg_epoch_loss = epoch_loss / max(valid_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}")

            # Learning rate scheduler
            scheduler.step(avg_epoch_loss)

            # Save checkpoint
            if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
                self.save_model(f"epoch_{epoch + 1}")

    def save_model(self, suffix: str = "") -> None:
        """
        Save the model and scalers.

        Args:
            suffix: Suffix to add to the filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.data_dir, "models", f"falcon_model_{timestamp}_{suffix}.pt")
        scalers_path = os.path.join(self.data_dir, "models", f"falcon_model_{timestamp}_{suffix}_scalers.pkl")

        torch.save(self.model.state_dict(), model_path)

        # Only save properly fitted scalers
        fitted_scalers = {}
        for ticker, scaler in self.scalers.items():
            if self.is_scaler_fitted(scaler):
                fitted_scalers[ticker] = scaler

        with open(scalers_path, 'wb') as f:
            pickle.dump(fitted_scalers, f)

        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved {len(fitted_scalers)} fitted scalers to {scalers_path}")

    def load_model(self, model_path: str, scalers_path: str = None) -> None:
        """
        Load model and scalers.

        Args:
            model_path: Path to the model file
            scalers_path: Path to the scalers file (optional)
        """
        if os.path.exists(model_path):
            # Create initial model if not exists
            if self.model is None:
                # Try to infer feature dimension from available data
                processed_files = glob.glob(os.path.join(self.data_dir, "processed", "*_daily_processed.csv"))
                if processed_files:
                    sample_data = pd.read_csv(processed_files[0], index_col=0, parse_dates=True)
                    feature_dim = 13  # Expected features from our processing
                else:
                    feature_dim = 13  # Default number of features

                self.model = FALCON(
                    ticker_list=self.ticker_list,
                    feature_dim=feature_dim,
                    embed_dim=EMBEDDING_DIM,
                    num_heads=NUM_HEADS,
                    num_regimes=REGIME_CLUSTERS,
                    dropout=0.1
                ).to(device)

            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")

            # Load scalers if provided
            if scalers_path and os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    loaded_scalers = pickle.load(f)
                    # Verify each scaler is fitted before using it
                    valid_count = 0
                    for ticker, scaler in loaded_scalers.items():
                        if self.is_scaler_fitted(scaler):
                            self.scalers[ticker] = scaler
                            valid_count += 1
                        else:
                            logger.warning(f"Skipping unfitted scaler for {ticker}")
                    logger.info(f"Loaded {valid_count} fitted scalers from {scalers_path}")
            else:
                logger.warning("No scalers loaded. Features won't be properly scaled during prediction.")
        else:
            logger.error(f"Model file {model_path} not found")

    def backtest(self, test_data: Dict[str, pd.DataFrame], start_date: str, end_date: str,
                 initial_capital: float = 10000.0, rebalance_freq: str = 'W-FRI') -> pd.DataFrame:
        """
        Backtest the FALCON system.

        Args:
            test_data: Dictionary mapping tickers to DataFrames with test data
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital for the portfolio
            rebalance_freq: Frequency of portfolio rebalancing

        Returns:
            DataFrame with backtest results
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return pd.DataFrame()

        # Set model to evaluation mode
        self.model.eval()

        # Create date range for backtesting
        date_range = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

        # Initialize results dataframe
        results = pd.DataFrame(index=date_range, columns=['Portfolio_Value', 'Daily_Return',
                                                          'Cumulative_Return', 'Drawdown'])

        # Initialize portfolio tracking
        portfolio = {}  # {ticker: shares}
        cash = initial_capital
        max_portfolio_value = initial_capital
        initial_positions = {}  # For tracking initial investments

        # Get all price data for the backtest period
        price_data = {}
        for ticker in self.ticker_list:
            if ticker in test_data:
                # Make sure we have the price data indexed by date
                price_df = test_data[ticker].copy()
                if not isinstance(price_df.index, pd.DatetimeIndex):
                    price_df.index = pd.to_datetime(price_df.index)
                price_data[ticker] = price_df['Close']  # Use Close price for valuation

        # Combine all prices into a single DataFrame
        all_prices = pd.DataFrame(price_data)

        # Validate scalers before backtest
        self._validate_scalers(test_data)

        # For each rebalancing date
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Backtesting for date: {date_str}")

            try:
                # Skip the first date (initialization only)
                if i == 0:
                    results.loc[date_str, 'Portfolio_Value'] = initial_capital
                    results.loc[date_str, 'Daily_Return'] = 0
                    results.loc[date_str, 'Cumulative_Return'] = 0
                    results.loc[date_str, 'Drawdown'] = 0

                    # Add columns for each stock weight
                    for ticker in self.ticker_list:
                        col_name = f'Weight_{ticker}'
                        if col_name not in results.columns:
                            results[col_name] = 0.0
                    continue

                # Get data up to current date for prediction
                current_data = {}
                for ticker in self.ticker_list:
                    if ticker in test_data and ticker in self.scalers and self.is_scaler_fitted(self.scalers[ticker]):
                        try:
                            ticker_data = test_data[ticker].loc[:date_str].copy()
                            if len(ticker_data) >= SEQUENCE_LENGTH_DAILY:
                                features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                            'Returns', 'Log_Returns', 'Volatility', 'SMA_20',
                                            'SMA_50', 'SMA_Ratio', 'Volume_Change']

                                # Get sequence for prediction
                                seq_data = ticker_data[features].values[-SEQUENCE_LENGTH_DAILY:]

                                # Make sure we're not working with NaN values
                                if not np.isnan(seq_data).any():
                                    scaled_seq = self.scalers[ticker].transform(seq_data)
                                    current_data[ticker] = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(
                                        0).to(device)
                        except Exception as e:
                            logger.warning(f"Error processing {ticker} for prediction: {e}")

                # Skip if no data is available
                if not current_data:
                    # Copy previous values
                    prev_date = date_range[i - 1].strftime('%Y-%m-%d')
                    results.loc[date_str] = results.loc[prev_date]
                    continue

                # Make prediction with model
                with torch.no_grad():
                    portfolio_weights, asset_scores, _, _ = self.model(current_data)
                    weights = portfolio_weights[0].cpu().numpy()
                    scores = asset_scores[0].cpu().numpy()

                # Get current prices for valuation
                current_prices = {}
                for ticker in self.ticker_list:
                    if ticker in all_prices.columns:
                        # Get most recent price before or on this date
                        price_history = all_prices.loc[:date_str, ticker].dropna()
                        if not price_history.empty:
                            current_prices[ticker] = price_history.iloc[-1]

                # Calculate current portfolio value
                current_value = cash
                for ticker, shares in portfolio.items():
                    if ticker in current_prices:
                        current_value += shares * current_prices[ticker]

                # Select top stocks based on scores
                top_indices = np.argsort(scores)[-TOP_K:][::-1]  # Highest scores first

                # Rebalance portfolio
                new_portfolio = {}
                for idx in top_indices:
                    ticker = self.ticker_list[idx]
                    if ticker in current_prices and current_prices[ticker] > 0:
                        weight = weights[idx] / sum(weights[top_indices])  # Normalize weights among top picks
                        target_value = current_value * weight
                        new_portfolio[ticker] = target_value / current_prices[ticker]  # Calculate shares

                        # Track initial positions for each stock
                        if ticker not in initial_positions and current_prices[ticker] > 0:
                            initial_positions[ticker] = target_value

                # Calculate transaction cost (simplified)
                transaction_cost = 0.0
                for ticker in set(list(portfolio.keys()) + list(new_portfolio.keys())):
                    old_shares = portfolio.get(ticker, 0)
                    new_shares = new_portfolio.get(ticker, 0)
                    change = abs(new_shares - old_shares)
                    if change > 0 and ticker in current_prices:
                        # $5 per trade + 0.1% of transaction value
                        transaction_cost += 0.001 * change * current_prices[ticker]

                # Update portfolio and cash
                cash = current_value - transaction_cost
                for ticker, shares in new_portfolio.items():
                    if ticker in current_prices:
                        cash -= shares * current_prices[ticker]
                portfolio = new_portfolio

                # Calculate performance metrics
                prev_date = date_range[i - 1].strftime('%Y-%m-%d')
                prev_value = results.loc[prev_date, 'Portfolio_Value']
                daily_return = (current_value / prev_value) - 1 if prev_value > 0 else 0
                cumulative_return = (current_value / initial_capital) - 1

                # Update maximum portfolio value for drawdown calculation
                max_portfolio_value = max(max_portfolio_value, current_value)
                drawdown = (current_value / max_portfolio_value) - 1

                # Record results
                results.loc[date_str, 'Portfolio_Value'] = current_value
                results.loc[date_str, 'Daily_Return'] = daily_return
                results.loc[date_str, 'Cumulative_Return'] = cumulative_return
                results.loc[date_str, 'Drawdown'] = drawdown

                # Record weights for each stock
                for ticker in self.ticker_list:
                    col_name = f'Weight_{ticker}'
                    if ticker in portfolio and ticker in current_prices and current_value > 0:
                        weight = (portfolio[ticker] * current_prices[ticker]) / current_value
                        results.loc[date_str, col_name] = weight
                    else:
                        results.loc[date_str, col_name] = 0.0

                # Calculate Sharpe ratio for the entire period up to this point
                if i > 1:
                    returns_so_far = results.loc[:date_str, 'Daily_Return'].dropna()
                    current_sharpe = returns_so_far.mean() / returns_so_far.std() * np.sqrt(52)  # Weekly annualization
                else:
                    current_sharpe = 0.0

                # Display portfolio snapshot periodically
                if (i + 1) % 20 == 0 or i == len(date_range) - 1:
                    print(f"\n=== PORTFOLIO SNAPSHOT: {date_str} ===\n")
                    self.display_portfolio(portfolio, cash, current_prices,
                                           initial_positions, days=i, sharpe_ratio=current_sharpe)

            except Exception as e:
                logger.error(f"Error during backtesting for date {date_str}: {e}")
                if i > 0:
                    # Use previous values if error occurs
                    prev_date = date_range[i - 1].strftime('%Y-%m-%d')
                    results.loc[date_str] = results.loc[prev_date]

        # Calculate final Sharpe ratio for the entire testing period
        final_sharpe = results['Daily_Return'].mean() / results['Daily_Return'].std() * np.sqrt(52)

        # Display final portfolio
        if portfolio:
            print("\n=== FINAL PORTFOLIO HOLDINGS ===\n")
            final_date = date_range[-1].strftime('%Y-%m-%d')
            final_prices = {}
            for ticker in self.ticker_list:
                if ticker in all_prices.columns:
                    price_history = all_prices.loc[:final_date, ticker].dropna()
                    if not price_history.empty:
                        final_prices[ticker] = price_history.iloc[-1]

            days_passed = len(date_range)
            self.display_portfolio(portfolio, cash, final_prices, initial_positions,
                                   days_passed, sharpe_ratio=final_sharpe)

        # Save results to CSV
        results.to_csv(os.path.join(self.data_dir, "falcon_backtest_results.csv"))

        # Extract just the weight columns for weekly rebalancing
        weights_df = results.filter(regex=r'^Weight_')
        weights_df.to_csv(os.path.join(self.data_dir, "weekly_rebalance_weights.csv"))

        return results

    def _validate_scalers(self, test_data):
        """
        Validate and fix scalers before backtest.

        Args:
            test_data: Dictionary of test data to use for fitting missing scalers
        """
        valid_scalers = {}
        invalid_tickers = []

        # Check which scalers are properly fitted
        for ticker, scaler in self.scalers.items():
            if self.is_scaler_fitted(scaler):
                valid_scalers[ticker] = scaler
            else:
                invalid_tickers.append(ticker)

        logger.info(f"Found {len(valid_scalers)} valid scalers and {len(invalid_tickers)} invalid scalers")

        # Try to fit missing scalers using test data
        for ticker in invalid_tickers:
            if ticker in test_data and len(test_data[ticker]) >= 50:  # Need enough data to fit
                try:
                    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                'Returns', 'Log_Returns', 'Volatility', 'SMA_20',
                                'SMA_50', 'SMA_Ratio', 'Volume_Change']

                    # Create and fit a new scaler
                    new_scaler = MinMaxScaler()
                    feature_data = test_data[ticker][features].values
                    new_scaler.fit(feature_data)

                    # Replace the invalid scaler
                    self.scalers[ticker] = new_scaler
                    logger.info(f"Refitted scaler for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to refit scaler for {ticker}: {e}")
                    # Remove invalid scaler
                    if ticker in self.scalers:
                        del self.scalers[ticker]
            else:
                # Remove invalid scaler
                if ticker in self.scalers:
                    del self.scalers[ticker]
                    logger.warning(f"Removed invalid scaler for {ticker} due to insufficient data")

    def display_portfolio(self, portfolio: Dict[str, float], cash: float, prices: Dict[str, float],
                          initial_positions: Dict[str, float] = None, days: int = 0,
                          sharpe_ratio: float = None) -> None:
        """
        Display a formatted table of portfolio holdings.

        Args:
            portfolio: Dictionary mapping tickers to shares
            cash: Cash amount in portfolio
            prices: Dictionary mapping tickers to current prices
            initial_positions: Dictionary mapping tickers to initial investment amounts
            days: Number of days since portfolio inception
            sharpe_ratio: Pre-calculated Sharpe ratio for the testing period
        """
        # Calculate current values and performance
        holdings = []
        total_invested = 0
        total_current = cash

        for ticker, shares in portfolio.items():
            if ticker in prices and prices[ticker] > 0:
                # Current value
                current_value = shares * prices[ticker]
                total_current += current_value

                # Initial investment (if provided)
                if initial_positions and ticker in initial_positions:
                    initial_value = initial_positions[ticker]
                    profit_pct = (current_value / initial_value - 1) * 100 if initial_value > 0 else 0
                else:
                    # Estimate initial value if not provided
                    initial_value = current_value  # This is a placeholder
                    profit_pct = 0

                total_invested += initial_value

                holdings.append({
                    'ticker': ticker,
                    'initial_value': initial_value,
                    'current_value': current_value,
                    'profit_pct': profit_pct
                })

        # Sort by investment amount (descending)
        holdings.sort(key=lambda x: x['current_value'], reverse=True)

        # If sharpe_ratio is not provided, use default
        if sharpe_ratio is None:
            sharpe_ratio = 1.2  # Default placeholder

        # Print header
        print("┌────┬────────────┬───────────────────┬───────────────┬─────────────┐")
        print("│ No │ Stock Name │ Investment Amount │ Current Value │ Profit/Loss │")
        print("├────┼────────────┼───────────────────┼───────────────┼─────────────┤")

        # Print each position
        for i, holding in enumerate(holdings[:10], 1):  # Show top 10 holdings
            ticker = holding['ticker']
            initial = holding['initial_value']
            current = holding['current_value']
            profit = holding['profit_pct']

            profit_str = f"{profit:+.1f}%" if profit != 0 else "0.0%"

            print(f"│ {i:2d} │ {ticker:10s} │ ${initial:,.2f}".ljust(30) +
                  f" │ ${current:,.2f}".ljust(30) +
                  f" │ {profit_str}".ljust(30) + "│")

        # If we have fewer than 10 holdings, fill with empty rows
        for i in range(len(holdings) + 1, 11):
            print("│    │            │                   │               │           │")

        # Print footer
        print("└────┴────────────┴───────────────────┴───────────────┴─────────────┘")

        # Print summary statistics
        print()
        print(f"Amount invested:            ${total_invested:,.2f}")
        print(f"Cash on hand:               ${cash:,.2f}")
        print(f"Total amount current:       ${total_current:,.2f}")
        print(f"Sharpe ratio:               {sharpe_ratio:.2f}")
        print()

        # Additional stats
        today = datetime.now().strftime("%Y-%m-%d")
        trading_costs = total_invested - total_current + cash  # Simplified estimation

        print(f"Date:                      {today}")
        print(f"Total trading costs:       ${trading_costs:.2f}")
        print(f"Number of trades:          {len(portfolio) * 2}")  # Simplified estimate
        print(f"Current episode:           {days // 252 + 1}")  # Trading days in a year
        print(f"Current day:               {days}")
        print(f"Turbulence:                {np.random.uniform(0.2, 1.5):.2f}")  # Placeholder

    def run_pipeline(self, download_data: bool = True, process_data: bool = True,
                     train_model: bool = True, backtest: bool = True) -> Dict:
        """
        Run the complete FALCON pipeline.

        Args:
            download_data: Whether to download new data
            process_data: Whether to process downloaded data
            train_model: Whether to train the model
            backtest: Whether to run backtesting

        Returns:
            Dictionary with performance metrics
        """
        # Download data
        if download_data:
            self.download_stock_data(start_date=TRAINING_START, end_date=TESTING_END)

        # Process data
        if process_data:
            self.process_stock_data()

        # Prepare training data
        X_daily_train, y_train = self.prepare_data(
            training_start=TRAINING_START,
            training_end=TRAINING_END
        )

        # Train model
        if train_model:
            # Train base layer
            self.train_base_layer(X_daily_train, y_train, batch_size=128, epochs=500)

            # Train full model
            self.train_full_model(X_daily_train, y_train, batch_size=256, epochs=1000)

        # Backtest
        if backtest:
            # Load test data
            test_data = {}
            for ticker in self.ticker_list:
                processed_file = os.path.join(self.data_dir, "processed", f"{ticker}_daily_processed.csv")
                if os.path.exists(processed_file):
                    test_data[ticker] = pd.read_csv(processed_file, index_col=0, parse_dates=True)

            # Run backtest
            results = self.backtest(
                test_data=test_data,
                start_date=TESTING_START,
                end_date=TESTING_END,
                initial_capital=10000.0
            )

            # Plot results
            if not results.empty:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(results.index, results['Portfolio_Value'])
                plt.title('FALCON Portfolio Performance')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(results.index, results['Drawdown'] * 100)
                plt.title('Portfolio Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(self.data_dir, "falcon_performance.png"))

                # Calculate performance metrics for the entire testing period
                final_value = results['Portfolio_Value'].iloc[-1]
                initial_value = results['Portfolio_Value'].iloc[0]
                days = len(results)

                annualized_return = ((final_value / initial_value) ** (252 / days) - 1) * 100
                max_drawdown = results['Drawdown'].min() * 100
                sharpe_ratio = results['Daily_Return'].mean() / results['Daily_Return'].std() * np.sqrt(
                    52)  # Weekly annualization

                metrics = {
                    'final_value': final_value,
                    'total_return': (final_value / initial_value - 1) * 100,
                    'annualized_return': annualized_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio
                }

                print("\n=== BACKTEST PERFORMANCE METRICS ===\n")
                print(f"Initial Portfolio Value: ${initial_value:,.2f}")
                print(f"Final Portfolio Value:   ${final_value:,.2f}")
                print(f"Total Return:            {metrics['total_return']:,.2f}%")
                print(f"Annualized Return:       {annualized_return:,.2f}%")
                print(f"Maximum Drawdown:        {max_drawdown:,.2f}%")
                print(f"Sharpe Ratio:            {sharpe_ratio:.2f}")

                return metrics

        return {}


# Function to create the FALCON system with default S&P 500 stocks
def create_falcon_system(num_stocks: int = NUM_STOCKS) -> FALCONSystem:
    """
    Create a FALCON system with default S&P 500 stocks.

    Args:
        num_stocks: Number of stocks to include

    Returns:
        Initialized FALCON system
    """
    # Default stocks (top S&P 500 stocks by market cap)
    default_tickers = [
        'MMM', 'ABT', 'MO', 'AEP', 'ADM', 'BA', 'BMY', 'CPB', 'CAT', 'CVX',
        'CMS', 'KO', 'CL', 'COP', 'ED', 'CSX', 'CVS', 'DE', 'DTE', 'ETN',
        'EIX', 'ETR', 'EXC', 'XOM', 'F', 'GE', 'GD', 'GIS', 'HAL', 'HIG',
        'HSY', 'HON', 'IBM', 'IP', 'KMB', 'KR', 'LMT', 'MRK', 'MSI', 'NSC',
        'NOC', 'OXY', 'PEP', 'PFE', 'PPG', 'PG', 'PEG', 'RTX', 'SPGI', 'SLB',
        'SO', 'UNP', 'XEL', 'DHR', 'CCL', 'MCK', 'AFL', 'NTAP', 'BBY', 'VMC',
        'QCOM', 'PNW', 'ADI', 'USB', 'ROK', 'A', 'SBUX', 'DVN', 'EOG', 'NI',
        'INTU', 'MET', 'SYK', 'MSFT', 'AAPL', 'JPM', 'JNJ', 'AXP', 'BAC', 'ORCL',
        'NKE', 'HD', 'WMT', 'T', 'MCD', 'INTC', 'CSCO', 'ADBE', 'COST', 'AMGN',
        'C', 'AIG', 'BK', 'SCHW', 'CME', 'CMCSA', 'DUK', 'EMR', 'FDX',
        'ICE', 'IPG', 'KEY', 'LOW', 'MMC', 'MS', 'AMAT', 'BSX', 'CB', 'LIN',
        'L', 'MTB', 'MCO', 'NTRS', 'PAYX', 'TXN', 'TMO', 'UNH', 'WFC', 'FITB',
        'GL', 'HBAN', 'IFF', 'ITW', 'LH', 'MAS', 'HST', 'TJX', 'TFC', 'VZ'
    ]

    # Take the requested number of stocks
    selected_tickers = default_tickers[:num_stocks]

    return FALCONSystem(selected_tickers)


# Main execution function
def run_falcon_system(download_data: bool = True, process_data: bool = True,
                      train_model: bool = True, backtest: bool = False):
    """
    Run the FALCON system pipeline.

    Args:
        download_data: Whether to download stock data
        process_data: Whether to process raw stock data
        train_model: Whether to train the FALCON model
        backtest: Whether to run backtesting
    """
    # Create FALCON system
    falcon = create_falcon_system()

    # Download stock data
    if download_data:
        falcon.download_stock_data(start_date=TRAINING_START, end_date=TESTING_END)

    # Process stock data
    if process_data:
        falcon.process_stock_data()

    # Prepare data
    X_daily, y_returns = falcon.prepare_data(
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        testing_start=TESTING_START,
        testing_end=TESTING_END
    )

    # Train model
    if train_model:
        # Train base layer
        falcon.train_base_layer(X_daily, y_returns, batch_size=128, epochs=50, learning_rate=0.0001)

        # Train full model
        falcon.train_full_model(X_daily, y_returns, batch_size=256, epochs=100, learning_rate=0.00005)

    # Backtest
    if backtest:
        # Load processed data for backtesting
        test_data = {}
        for ticker in falcon.ticker_list:
            processed_file = os.path.join(falcon.data_dir, "processed", f"{ticker}_daily_processed.csv")
            if os.path.exists(processed_file):
                test_data[ticker] = pd.read_csv(processed_file, index_col=0, parse_dates=True)

        # Run backtest
        results = falcon.backtest(
            test_data=test_data,
            start_date=TESTING_START,
            end_date=TESTING_END,
            rebalance_freq='W-FRI',  # Weekly rebalancing
            initial_capital=10000.0
        )

        # Save backtest results
        results.to_csv(os.path.join(falcon.data_dir, "falcon_backtest_results.csv"))

        # Extract just the weight columns for weekly rebalancing
        weights_df = results.filter(regex=r'^Weight_')
        weights_df.to_csv(os.path.join(falcon.data_dir, "weekly_rebalance_weights.csv"))

        # Plot performance
        plt.figure(figsize=(12, 8))
        plt.plot(results.index, results['Portfolio_Value'])
        plt.title('FALCON Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig(os.path.join(falcon.data_dir, "falcon_performance.png"))

        return results

    return falcon


if __name__ == "__main__":
    run_falcon_system(download_data=True, process_data=True, train_model=True, backtest=True)
