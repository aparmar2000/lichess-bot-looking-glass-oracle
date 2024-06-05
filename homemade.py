"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import logging
import random
from typing import Optional

import chess
from chess.engine import Limit, PlayResult
import torch

from lib import model
from lib.config import Configuration
from lib.engine_wrapper import MinimalEngine
from lib.types import COMMANDS_TYPE, HOMEMADE_ARGS_TYPE, MOVE, OPTIONS_GO_EGTB_TYPE
from looking_glass_bot.mamba_chess_model import MambaChessModel, load_from_safetensors
from transformers import MambaConfig

from looking_glass_bot.mamba_chess_utils import mamba_score_moves

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)

class LookingGlassEngine(ExampleEngine):
    
    mamba_model: MambaChessModel | None = None

    def __init__(self, commands: COMMANDS_TYPE, options: OPTIONS_GO_EGTB_TYPE, stderr: Optional[int],
                draw_or_resign: Configuration, game: Optional[model.Game] = None, name: Optional[str] = None,
                **popen_args: str) -> None:
        super().__init__(commands, options, stderr, draw_or_resign, game, name, **popen_args)
        
        my_elo = 3000
        opponent_elo = 1500
        if game is not None:
            if game.opponent.rating is not None:
                opponent_elo = game.opponent.rating
            elif game.opponent.is_bot:
                opponent_elo = 3000
            
        if game is None or game.is_white:
            self.player_1_elo = my_elo
            self.player_2_elo = opponent_elo
        else:
            self.player_1_elo = opponent_elo
            self.player_2_elo = my_elo
        
        if LookingGlassEngine.mamba_model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = 'cpu'
            
            configuration = MambaConfig(
                vocab_size=85,
                hidden_size=256,
                num_hidden_layers=6
            )
            LookingGlassEngine.mamba_model = load_from_safetensors(
                configuration,
                "looking_glass_bot/models/model_6x256.safetensors",
                device=device
            )
            LookingGlassEngine.mamba_model.device = device
        
    
    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        assert LookingGlassEngine.mamba_model is not None
        
        scored_moves = mamba_score_moves(board, LookingGlassEngine.mamba_model, self.player_1_elo, self.player_2_elo)
        
        scored_moves.sort(reverse=True, key=lambda m: m.score)
        
        logger.info(
            "Top 5 moves: [%s]", 
            ','.join([scored.move.uci()+' '+str(scored.score) for scored in scored_moves[:5] ])
            )
        
        return PlayResult(scored_moves[0].move, None, draw_offered=draw_offered)