"""Self-learning: neural nets (micro + deep), transformer, encoder, Q-learning."""
from .adam import AdamOptimizer
from .deep_net import DeepNet
from .encoder import HashingEncoder
from .hybrid_next_token import fluency, predict_next
from .learner import LearningModule
from .micro_net import MicroNet
from .next_token import NextTokenModel, TokenPrediction
from .qlearn import QLearningAgent
from .transformer import TransformerConfig, TransformerLM, TransformerTrainingReport

__all__ = [
    "AdamOptimizer", "DeepNet", "HashingEncoder", "LearningModule", "MicroNet",
    "NextTokenModel", "QLearningAgent", "TokenPrediction", "TransformerConfig",
    "TransformerLM", "TransformerTrainingReport", "fluency", "predict_next",
]
