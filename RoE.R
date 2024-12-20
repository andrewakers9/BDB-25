library(tidyverse)

play_df <- read_csv("data/plays_df", col_type = cols()) %>%
  janitor::clean_names()
player_play_df <- read_csv("data/player_play.csv", col_types = cols()) %>%
  janitor::clean_names()

label_df <- read_csv("data/labels.csv", col_types = cols())
cv_blitz_preds <- read_csv("data/cv_preds.csv")

pass_play_ids <- plays_df %>%
  filter(is_dropback)

# remove non-pass plays
plays_df <- plays_df %>%
  semi_join(
    pass_play_ids,
    by = c("game_id", "play_id")
  )
player_play_df <- player_play_df %>%
  semi_join(
    pass_play_ids,
    by = c("game_id", "play_id")
  )

cv_blitz_preds <- cv_blitz_preds %>%
  left_join(
    label_df %>%
      select(game_id, play_id, player_id, nfl_id),
    by = c("game_id", "play_id", "player_id")
  )

pass_rush_play_df <- player_play_df %>%
  left_join(
    label_df %>%
      select(game_id, play_id, nfl_id, y),
    by = c("game_id", "play_id", "nfl_id")
  ) %>%
  mutate(
    is_blitzer = if_else(y == 1, 1, 0)
  ) %>%
  group_by(game_id, play_id) %>%
  summarise(
    n_rushers = sum(was_initial_pass_rusher, na.rm = TRUE),
    n_blitzers = sum(is_blitzer, na.rm = TRUE),
    olb_blitzers = sum(is_blitzer * as.numeric(position == "OLB")),
    ilb_blitzers = sum(is_blitzer * as.numeric(position == "ILB")),
    ss_blitzers = sum(is_blitzer * as.numeric(position == "SS")),
    fs_blitzers = sum(is_blitzer * as.numeric(position == "FS")),
    cb_blitzers = sum(is_blitzer * as.numeric(position == "CB")),
    n_blockers = 10 - sum(was_running_route, na.rm = TRUE),
    pressure = as.numeric(any(caused_pressure == 1)),
    .groups = "drop"
  ) 

expected_rushers <- data.frame()
tts_seq <- seq(0, 100, by = 10)
for(tts in tts_seq) {
  tmp <- label_df %>%
    left_join(
      cv_blitz_preds %>%
        filter(time_to_snap == tts) %>%
        select(game_id, play_id, player_id, pred),
      by = c("game_id", "play_id", "player_id")
    ) %>%
    group_by(game_id, play_id, y) %>%
    summarise(
      n = n(),
      n_pass_rushers = sum(pass_rusher),
      roe = sum(1 - pred),
      doe = sum(pred),
      .groups = "drop"
    )
  tmp <- tmp %>%
    group_by(game_id, play_id) %>%
    filter(
      any(y == 1) | sum(n_pass_rushers) > 4
    ) %>%
    ungroup() %>%
    mutate(
      roe = if_else(y != 1, 0, roe),
      doe = case_when(
        y == 1 ~ 0,
        y == -1 ~ n - n_pass_rushers,
        y == 0 ~ doe
      )
    )
  tmp <- tmp %>%
    group_by(game_id, play_id) %>%
    summarise(
      n_pass_rushers = sum(n_pass_rushers),
      roe = sum(roe),
      doe = sum(doe),
      .groups = "drop"
    ) %>%
    filter(!is.na(roe))
  tmp <- tmp %>%
    left_join(
      pass_rush_play_df %>%
        select(game_id, play_id, n_blitzers, pressure),
      by = c("game_id", "play_id")
    ) %>%
    mutate(
      avg_roe = coalesce(roe / n_blitzers, 0),
      doe = replace_na(doe, 0),
      time_to_snap = tts
    ) %>%
    relocate(time_to_snap, .after = play_id)
  
  expected_rushers <- bind_rows(expected_rushers, tmp)
}
expected_rushers <- expected_rushers %>%
  arrange(game_id, play_id, time_to_snap)
  
get_log_reg_output <- function(data) {
  mod <- glm(pressure ~ avg_roe, family = "binomial", data = data)
  out <- broom::tidy(mod)
  return(out)
}
log_reg_output <- expected_rushers %>%
  nest(.by = time_to_snap) %>%
  mutate(log_reg_output = map(data, get_log_reg_output)) %>%
  select(time_to_snap, log_reg_output) %>%
  unnest(log_reg_output) 

avg_roe_coefs <- log_reg_output %>%
  filter(term == "avg_roe") %>%
  mutate(
    upper = estimate + 1.96 * std.error,
    lower = estimate - 1.96 * std.error
  )

avg_roe_coefs %>%
  mutate(
    time_to_snap = time_to_snap / 10,
    time_to_snap = factor(time_to_snap, levels = rev(seq(0, 10, by = 1))) 
  ) %>%
  ggplot(
    aes(factor(time_to_snap), estimate)
  ) + 
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.2
  ) + 
  scale_y_continuous(
    name = "Avg RoE Regression Coefficient",
    limits = c(0, 1.25),
    breaks = seq(0, 1.25, by = 0.25)
  ) + 
  xlab("Time to Snap (Seconds)") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank()
  )
 ggsave("visualizations/avg_roe_coefs.png", 
        dpi = 500,
        height = 4,
        width = 7) 
