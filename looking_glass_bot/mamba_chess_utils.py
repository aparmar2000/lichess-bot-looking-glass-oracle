import typing
import chess
import torch
import torch.nn.functional as F

from looking_glass_bot.mamba_chess_model import MambaChessModel

def token_set_to_move(board: chess.Board, token_set) -> chess.Move:
    promotion_piece_type = None
    if token_set[0] != token_set[3]:
        promotion_piece_type = token_set[3]-64
        if promotion_piece_type>chess.KING:
            promotion_piece_type -= chess.KING
    
    return board.find_move(
        from_square=token_set[1],
        to_square=token_set[2],
        promotion=promotion_piece_type
    )

def tokenize_piece(piece: chess.Piece) -> int:
    piece_token_id = piece.piece_type
    piece_token_id += chess.KING if piece.color else 0
    piece_token_id += 64
    return piece_token_id

def tokenize_move(board: chess.Board, move: chess.Move) -> tuple[int, int, int, int]:
    piece_moved = board.piece_at(move.from_square)
    if piece_moved is None:
        raise LookupError("No piece at source position??")
    from_piece_token_id = tokenize_piece(piece_moved)
    to_piece_token_id = tokenize_piece( chess.Piece(move.promotion,piece_moved.color) ) if move.promotion else from_piece_token_id
    return (from_piece_token_id, move.from_square, move.to_square, to_piece_token_id)

# Move ranking methods

ScoredMove = typing.NamedTuple('ScoredMove', [('move', chess.Move), ('score', float)])
def mamba_score_moves(board: chess.Board, model: MambaChessModel, player_1_elo=1500, player_2_elo=1500) -> list[ScoredMove]:
    board_root_copy = board.root()
    move_tokens = []
    for move in [*board.move_stack]:
        move_tokens.extend(tokenize_move(board_root_copy, move))
        board_root_copy.push(move)
    move_tokens = [t+1 for t in move_tokens]
    
    move_token_queue = {}
    for legal_move in board.legal_moves:
        legal_token_seq = []
        for token in tokenize_move(board, legal_move):
            adj_token = token+1
            
            seq_key = ",".join([str(t) for t in legal_token_seq])
            if seq_key not in move_token_queue:
                step_move_tokens = move_tokens+legal_token_seq
                step_move_tokens = step_move_tokens[-model.move_context_size()*4:]
                token_ids = torch.tensor([78,79,80,  81,82,83]+step_move_tokens, dtype=torch.int)
                token_ids = F.pad(token_ids, (model.token_context_size()-len(token_ids),0), "constant", 0)
                
                move_token_queue[seq_key] = (set(), token_ids)
            move_token_queue[seq_key][0].add(adj_token)
            legal_token_seq.append(adj_token)
    
    player_1_elo_tensor = torch.tensor( [player_1_elo] ).float()
    player_2_elo_tensor = torch.tensor( [player_2_elo] ).float()
    unmap_map:dict[int, str] = {i:k for i,k in enumerate(move_token_queue.keys())}
    logits = model(
        torch.stack( [move_token_queue[unmap_map[i]][1] for i in unmap_map.keys()] ).to(model.device, dtype=torch.int),
        player_1_elo=player_1_elo_tensor.to(model.device),
        player_2_elo=player_2_elo_tensor.to(model.device)
    )['logits'][0][:,-1]
    next_token_probs = F.softmax(logits, dim=1)
    move_token_results: dict[str, dict[int, float]] = {}
    for i in range(len(next_token_probs)):
        key = unmap_map[i]
        target_tokens = move_token_queue[key][0]
        move_token_results[key] = {target_token:next_token_probs[i][target_token].item() for target_token in target_tokens}
    
    move_scores:list[ScoredMove] = []
    for legal_move in board.legal_moves:
        legal_token_seq:list[int] = []
        move_score = 1.0
        for token in tokenize_move(board, legal_move):
            adj_token = token+1

            seq_key = ",".join([str(t) for t in legal_token_seq])
            move_score *= move_token_results[seq_key][adj_token]

            legal_token_seq.append(adj_token)
        move_scores.append(ScoredMove(legal_move, move_score))

    return move_scores
