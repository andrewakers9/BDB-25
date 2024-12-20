library(gganimate)
library(ggnewscale)
library(tidyverse)

source("play-animation.R")

games_df <- read_csv("data/games.csv", col_types = cols()) %>%
  janitor::clean_names()
plays_df <- read_csv("data/plays.csv", col_types = cols()) %>%
  janitor::clean_names()
players_df <- read_csv("data/players.csv", col_types = cols()) %>%
  janitor::clean_names()

cv_blitz_preds <- read_csv("data/cv_preds.csv", col_types = cols())
label_df <- read_csv("data/labels.csv", col_types = cols())

game_id_ <- 2022103100
play_id_ <- 1171

week <- games_df %>%
  filter(game_id == game_id_) %>%
  pull(week)
tracking_week_df <- read_csv(
  paste0("data/tracking_week_", week, ".csv"),
  col_types = cols()
) %>%
  janitor::clean_names()

# mirror tracking data, so all plays are in the same direction
tracking_week_df <- tracking_week_df %>%
  mutate(
    x = if_else(play_direction == "left", 120 - x, x),
    y = if_else(play_direction == "left", 160/3 - y, y),
    o = case_when(
      play_direction == "left" & o > 180 ~ o - 180,
      play_direction == "left" & o < 180 ~ o + 180,
      TRUE ~ o
    ),
    dir = case_when(
      play_direction == "left" & dir > 180 ~ dir - 180,
      play_direction == "left" & dir < 180 ~ dir + 180,
      TRUE ~ dir
    )
  )

def_team <- plays_df %>%
  filter(
    game_id == game_id_,
    play_id == play_id_
  ) %>%
  pull(defensive_team)

play_df <- tracking_week_df %>%
  filter(
    game_id == game_id_,
    play_id == play_id_
  ) %>%
  left_join(
    players_df %>%
      select(nfl_id, position),
    by = "nfl_id"
  )

play_df <- play_df %>%
  arrange(club, nfl_id) %>%
  group_by(frame_id, club) %>%
  mutate(
    player_id = row_number() - 1
  ) %>%
  ungroup() %>%
  mutate(
    player_id = if_else(club == def_team, player_id, NA)
  )

play_df <- play_df %>%
  left_join(
    cv_blitz_preds %>%
      filter(game_id == game_id_, play_id == play_id_) %>%
      select(frame_id, player_id, blitz_prob = pred),
    by = c("frame_id", "player_id")
  ) %>%
  left_join(
    label_df %>%
      filter(game_id == game_id_, play_id == play_id_) %>%
      select(nfl_id, blitz = y),
    by = "nfl_id"
  )
animate_play(play_df, 
             nframes = 200, 
             show_path_ids = c(52473, 46146))
anim_save("visualizations/play_example.gif")
