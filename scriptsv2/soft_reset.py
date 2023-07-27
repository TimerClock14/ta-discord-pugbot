import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy
from dateutil.parser import parse as parse_date
from sqlalchemy import or_
from trueskill import Rating, rate
from typing_extensions import Literal

from discord_bots.models import Player, Session, QueueRegion, PlayerRegionTrueskill, FinishedGame, FinishedGamePlayer

level = logging.INFO


def define_logger(name="app"):
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s:%(filename)s:%(lineno)s] %(message)s")
    console_logger = logging.StreamHandler()
    console_logger.setLevel(level)
    console_logger.setFormatter(formatter)
    log.addHandler(console_logger)
    return log


log = define_logger('soft_reset')

OutcomeType = Literal["team1", "team2", "tie"]
default_rating = Rating()
DRAW = [0, 0]
TEAM1_WIN = [0, 1]
TEAM2_WIN = [1, 0]


@dataclass
class RawGame:
    team1: list[int]
    team2: list[int]
    outcome: OutcomeType
    rated: bool


@dataclass
class PlayerRating:
    id: int
    rated_mu: float
    rated_sigma: float
    unrated_mu: float
    unrated_sigma: float


def parse_args() -> dict[str, any]:
    parser = argparse.ArgumentParser(
        description="Soft reset a queue region. Defaults to 'dry run' unless specifically told to overwrite data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--store",
                        default="False",
                        help="If 'True' will store the recalculated ratings. "
                             "Defaults to 'False', which is a 'Dry Run' mode.")
    parser.add_argument("--src-regions",
                        nargs="*",
                        help="Regions to source finished games from. "
                             "At least one of --src-queues or --src-regions must be supplied")
    parser.add_argument("--src-queues",
                        nargs="*",
                        help="Queues to source finished games from, regardless of set region. "
                             "At least one of --src-queues or --src-regions must be supplied")
    parser.add_argument("--from",
                        help="Date to start rating games. Leave empty to use all games, "
                             "supply date in format YYYY-MM-DD or YYYY-MM-DDThh:mm:ss "
                             "to start fetching games fom that date at UTC 00:00 AM")
    parser.add_argument("--target-region",
                        help="Name of the queue region to calculate or recalculate",
                        required=True)
    arguments = parser.parse_args()
    return vars(arguments)


def map_raw_games(game_history: list[FinishedGame], game_players: list[FinishedGamePlayer]) -> list[RawGame]:
    output: list[RawGame] = []

    log.info(f"Started mapping {len(game_history)} games")
    for idx, game in enumerate(game_history):
        players = list(filter(lambda fgp: fgp.finished_game_id == game.id, game_players))
        players_team1 = list(filter(lambda fgp: fgp.team == 0, players))
        players_team2 = list(filter(lambda fgp: fgp.team == 1, players))
        if len(players_team1) != len(players_team2):
            log.warning(f"Ignoring game {game.game_id}. Player count not balanced.")
            continue
        # noinspection PyTypeChecker
        outcome: OutcomeType = "team1" if game.winning_team == 0 else ("team2" if game.winning_team == 1 else "tie")
        output.append(RawGame(team1=list(map(lambda fgp: fgp.player_id, players_team1)),
                              team2=list(map(lambda fgp: fgp.player_id, players_team2)),
                              outcome=outcome,
                              rated=game.is_rated))
        if idx % 1000 == 0 and idx > 0:
            log.info(f"Mapped {idx}/{len(game_history)} games")

    log.info(f"Finished mapping {len(game_history)} games")
    return output


def rate_games(games: list[RawGame]) -> dict[int, PlayerRating]:
    def get_player_or_default(players: dict[int, PlayerRating], player_id: int) -> PlayerRating:
        player = players.get(player_id)
        if player is None:
            player = PlayerRating(id=player_id, rated_mu=default_rating.mu, rated_sigma=default_rating.sigma,
                                  unrated_mu=default_rating.mu, unrated_sigma=default_rating.sigma)
            players.update({player_id: player})
        return player

    def player_to_rating(player: PlayerRating, rated: bool) -> Rating:
        return Rating(mu=player.rated_mu if rated else player.unrated_mu,
                      sigma=player.rated_sigma if rated else player.unrated_sigma)

    def update_ratings(team: list[PlayerRating], new_ratings: list[Rating], rated: bool):
        for update_idx, player in enumerate(team):
            if rated:
                player.rated_mu = new_ratings[update_idx].mu
                player.rated_sigma = new_ratings[update_idx].sigma
            else:
                player.unrated_mu = new_ratings[update_idx].mu
                player.unrated_sigma = new_ratings[update_idx].sigma

    result: dict[int, PlayerRating] = {}
    log.info(f"Started rating {len(games)} games")
    for idx, game in enumerate(games):
        team1 = list(map(lambda i: get_player_or_default(result, i), game.team1))
        team2 = list(map(lambda i: get_player_or_default(result, i), game.team2))
        team1_ratings_before = list(map(lambda p: player_to_rating(p, game.rated), team1))
        team2_ratings_before = list(map(lambda p: player_to_rating(p, game.rated), team2))
        game_result = TEAM1_WIN if game.outcome == "team1" else (TEAM2_WIN if game.outcome == "team2" else DRAW)
        team1_ratings_after, team2_ratings_after = rate([team1_ratings_before, team2_ratings_before], game_result)
        update_ratings(team1, team1_ratings_after, game.rated)
        update_ratings(team2, team2_ratings_after, game.rated)

        if idx % 1000 == 0 and idx > 0:
            log.info(f"Rated {idx}/{len(games)} games")

    log.info(f"Finished rating {len(games)} games")
    return result


def map_ratings_to_entities(session: Session, ratings: dict[int, PlayerRating], target_region_id: str) -> \
        list[tuple[Player, PlayerRegionTrueskill | None, PlayerRegionTrueskill | None]]:
    """
    map for update or display
    :param session: session object
    :param ratings: calculated ratings
    :param target_region_id: target region
    :return: player -> (old_rating, new_rating)
    """
    new_players: list[Player] = session.query(Player).filter(Player.id.in_(list(ratings.keys()))).all()
    old_ratings_result = session.query(PlayerRegionTrueskill, Player) \
        .filter(PlayerRegionTrueskill.player_id == Player.id,
                PlayerRegionTrueskill.queue_region_id == target_region_id) \
        .all()
    old_prts: list[PlayerRegionTrueskill] = [x[0] for x in old_ratings_result]
    old_players: list[Player] = [x[1] for x in old_ratings_result]
    all_players = numpy.unique(old_players + new_players)

    result: list[tuple[Player, PlayerRegionTrueskill | None, PlayerRegionTrueskill | None]] = []
    player: Player
    for player in all_players:
        old_prt = next(iter(filter(lambda prt: prt.player_id == player.id, old_prts)), None)
        new_prt = None
        rating = ratings.get(player.id)
        if rating is not None:
            new_prt = PlayerRegionTrueskill(player_id=rating.id,
                                            queue_region_id=target_region_id,
                                            rated_trueskill_mu=rating.rated_mu,
                                            rated_trueskill_sigma=rating.rated_sigma,
                                            unrated_trueskill_mu=rating.unrated_mu,
                                            unrated_trueskill_sigma=rating.unrated_sigma)
        result.append((player, old_prt, new_prt))
    return result


def print_ratings_change(
        new_rating_entries: list[tuple[Player, PlayerRegionTrueskill | None, PlayerRegionTrueskill | None]]) -> None:
    entries = new_rating_entries.copy()
    # sort by new rating, then old rating, then player-id
    entries.sort(
        key=lambda x: (
            x[2].leaderboard_trueskill if x[2] is not None else -1,
            x[1].leaderboard_trueskill if x[1] is not None else -1,
            x[0].id),
        reverse=True)
    output = "id;name;old_rating;new_rating;old_mu;new_mu;old_sigma;new_sigma\n"
    for entry in entries:
        player = entry[0]
        old_rating = entry[1]
        new_rating = entry[2]
        escaped_name = player.name.replace("\"", "\\")
        output += f'{player.id};"{escaped_name}";' \
                  f'{"" if old_rating is None else round(old_rating.leaderboard_trueskill, 1)};' \
                  f'{"" if new_rating is None else round(new_rating.leaderboard_trueskill, 1)};' \
                  f'{"" if old_rating is None else round(old_rating.rated_trueskill_mu, 1)};' \
                  f'{"" if new_rating is None else round(new_rating.rated_trueskill_mu, 1)};' \
                  f'{"" if old_rating is None else round(old_rating.rated_trueskill_sigma, 1)};' \
                  f'{"" if new_rating is None else round(new_rating.rated_trueskill_sigma, 1)}\n'
    log.info("### Changed Ratings:\n\n" + output + "\n\n ###")


def store_updated_ratings(session: Session,
                          new_rating_entries: list[
                              tuple[Player, PlayerRegionTrueskill | None, PlayerRegionTrueskill | None]],
                          target_region_id: str) -> None:
    session.query(PlayerRegionTrueskill) \
        .filter(PlayerRegionTrueskill.queue_region_id == target_region_id) \
        .delete()
    new_ratings = [x[2] for x in new_rating_entries if x[2] is not None]
    for new_rating in new_ratings:
        session.add(new_rating)


def do_soft_reset(target_region: str, src_regions: list[str], src_queues: list[str], from_date: datetime,
                  dry_run: bool) -> None:
    log.info(
        f"Executing soft reset: target regions {target_region}, source regions: {src_regions}, "
        f"source queues: {src_queues}, from: {from_date}, dry run: {dry_run}")
    if dry_run:
        log.info(f"Executing dry run")
    else:
        log.warning(f"Real mode. Data will be overwritten!")

    with Session() as session:
        target_region: QueueRegion | None = session.query(QueueRegion).filter(target_region == QueueRegion.name).first()
        if target_region is None:
            raise ValueError(f"Region {target_region} does not exist")

        log.info("Loading game history")
        # noinspection PyUnresolvedReferences
        game_history: list[FinishedGame] = session.query(FinishedGame) \
            .filter(or_(FinishedGame.queue_name.in_(src_queues), FinishedGame.queue_region_name.in_(src_regions)),
                    FinishedGame.finished_at >= from_date) \
            .order_by(FinishedGame.finished_at.asc()) \
            .all()
        # noinspection PyUnresolvedReferences
        game_players: list[FinishedGamePlayer] = session.query(FinishedGamePlayer) \
            .filter(FinishedGamePlayer.finished_game_id.in_(list(map(lambda x: x.id, game_history)))) \
            .all()
        log.info("Finished loading game history")

        games = map_raw_games(game_history, game_players)
        ratings = rate_games(games)
        new_rating_entries = map_ratings_to_entities(session, ratings, target_region.id)
        print_ratings_change(new_rating_entries)

        if not dry_run:
            store_updated_ratings(session, new_rating_entries, target_region.id)
            session.commit()


def main() -> None:
    input_args = parse_args()
    if input_args["src_queues"] is None and input_args["src_regions"] is None:
        log.error("At least one of --src-queues or --src-regions must be supplied")
        exit(1)

    target_region = input_args["target_region"]
    src_queues = input_args["src_queues"] if input_args["src_queues"] is not None else []
    src_regions = input_args["src_regions"] if input_args["src_regions"] is not None else []
    dry_run = False if input_args["store"].lower() == "true" else True
    from_date = parse_date(input_args["from"]) if input_args["from"] is not None else datetime(1990, 1, 1)

    do_soft_reset(
        target_region=target_region,
        src_regions=src_regions,
        src_queues=src_queues,
        from_date=from_date,
        dry_run=dry_run)


if __name__ == "__main__":
    main()
