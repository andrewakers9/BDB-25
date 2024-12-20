library(yardstick)
library(tidyverse)

players_df <- read_csv("data/players.csv", col_types = cols()) %>%
  janitor::clean_names()
label_df <- read_csv("data/labels.csv", col_type = cols())
cv_blitz_preds <- read_csv("data/cv_preds.csv", col_type = cols())

# group low frequency positions 
players_df <- players_df %>%
  mutate(
    position = case_when(
      position == "LB" ~ "DE", # only Josh Sweat 
      position == "DB" ~ "CB", # only Ronnie Harrison 
      position == "MLB" ~ "ILB",
      TRUE ~ position
    )
  )

cv_blitz_preds <- cv_blitz_preds %>%
  left_join(
    label_df %>%
      select(game_id, play_id, player_id, nfl_id),
    by = c("game_id", "play_id", "player_id")
  ) %>%
  left_join(
    players_df %>%
      select(nfl_id, position),
    by = "nfl_id"
  )

overall_performance <- bind_rows(
  cv_blitz_preds %>%
    mutate(
      y = factor(y),
      pred = factor(if_else(pred > 0.5, 1, 0))
    ) %>%
    recall(y, pred, event_level = "second"),
  cv_blitz_preds %>%
    mutate(
      y = factor(y),
      pred = factor(if_else(pred > 0.5, 1, 0))
    ) %>%
    precision(y, pred, event_level = "second"),
  cv_blitz_preds %>%
    mutate(
      y = factor(y)
    ) %>%
    roc_auc(y, pred, event_level = "second")
)

position_performance <- bind_rows(
  cv_blitz_preds %>%
    mutate(
      y = factor(y),
      pred = factor(if_else(pred > 0.5, 1, 0))
    ) %>%
    group_by(position) %>%
    recall(y, pred, event_level = "second"),
  cv_blitz_preds %>%
    mutate(
      y = factor(y),
      pred = factor(if_else(pred > 0.5, 1, 0))
    ) %>%
    group_by(position) %>%
    precision(y, pred, event_level = "second"),
  cv_blitz_preds %>%
    mutate(
      y = factor(y)
    ) %>%
    group_by(position) %>%
    roc_auc(y, pred, event_level = "second")
)

write_csv(overall_performance, "evaluation/overall_blitz_performance.csv")  
write_csv(position_performance, "evaluation/position_blitz_performance.csv")
