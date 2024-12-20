library(gt)
library(gtExtras)
library(nflfastR)
library(tidyverse)

plays_df <- read_csv("data/plays.csv", col_types = cols()) %>%
  janitor::clean_names()
player_play_df <- read_csv("data/player_play.csv", col_types = cols()) %>%
  janitor::clean_names()

cv_pressure_preds <- read_csv("data/cv_pressure_preds3.csv")

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

cv_pressure_preds <- cv_pressure_preds %>%
  group_by(game_id, play_id) %>%
  summarise(
    across(everything(), last),
    .groups = "drop"
  )

pass_rush_play_df <- player_play_df %>%
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
  ) %>%
  left_join(
    plays_df %>%
      mutate(
        sack = if_else(is.na(time_to_sack), 0, 1),
        time_to_event = coalesce(time_to_throw, time_to_sack, time_in_tackle_box)
      ) %>%
      select(game_id, play_id, sack, time_to_event, pff_man_zone, pff_pass_coverage),
    by = c("game_id", "play_id")
  )
baseline_pressure_rate <- pass_rush_play_df %>%
  filter(n_blitzers > 0 | n_rushers > 4) %>%
  group_by(n_rushers) %>%
  summarise(
    n = n(),
    baseline_pred = mean(pressure)
  )

cv_pressure_preds <- cv_pressure_preds %>%
  left_join(
    pass_rush_play_df %>%
      select(game_id, play_id, n_rushers, n_blitzers, pressure),
    by = c("game_id", "play_id")
  ) %>%
  left_join(
    baseline_pressure_rate %>% select(-n),
    by = "n_rushers"
  )

team_pressure <- cv_pressure_preds %>%
  mutate(
    pressure_rate_rel_baseline = pressure - baseline_pred,
    nprg = pred - baseline_pred
  ) %>%
  left_join(
    plays_df %>%
      select(game_id, play_id, defensive_team),
    by = c("game_id", "play_id")
  ) %>%
  group_by(defensive_team) %>%
  summarise(
    n = n(),
    avg_n_rushers = mean(n_rushers),
    pressure_rate = mean(pressure),
    exp_baseline_pressure_rate = mean(baseline_pred),
    exp_pressure_rate = mean(pred),
    pressure_rate_rel_baseline = mean(pressure_rate_rel_baseline),
    nprg = mean(nprg),
    .groups = "drop"
  )

## visualizations ##
cv_pressure_preds %>%
  mutate(
    pressure_label = if_else(pressure == 1, "Pressure", "No Pressure")
  ) %>%
  ggplot(
    aes(pred, fill = pressure_label)
  ) + 
  geom_density(alpha = 0.5) +
  scale_x_continuous(
    name = "Pressure Probability",
    limits = c(0, 1),
    labels = scales::percent
  ) +
  scale_fill_discrete(
    name = ""
  ) +
  theme_minimal()
ggsave("visualizations/pressure_dist.png",
       dpi = 500,
       height = 4,
       width = 7)

team_logos <- nflreadr::load_teams() %>%
  select(team_abbr, team_logo_wikipedia)

team_pressure <- team_pressure %>%
  left_join(
    team_logos %>%
      select(team_abbr, team_logo_wikipedia, team_color),
    by = join_by(defensive_team == team_abbr)
  )

# team logo scatter plot
team_pressure %>%
  ggplot(
    aes(nprg, pressure_rate_rel_baseline)
  ) + 
  geom_smooth(
    method = "lm",
    se = FALSE
  ) + 
  geom_image(
    aes(image = team_logo_wikipedia),
    size = 0.1
  ) + 
  geom_hline(
    yintercept = 0,
    alpha = 0.3
  ) + 
  geom_vline(
    xintercept = 0,
    alpha = 0.3
  ) +
  scale_y_continuous(
    name = "Pressure rate relative to baseline",
    labels = scales::percent
  ) +
  scale_x_continuous(
    name = "NPRG",
    labels = scales::percent,
    limits = c(-0.06, 0.06)
  ) + 
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(size = 12)
  ) +
  ggtitle(
    "Relationship between NPRG and pressure rate\nrelative to baseline (r = 0.44)"
  )
ggsave("visualizations/nprg_scatter.png", 
       dpi = 500, 
       device = "png",
       height = 4,
       width = 5)

# team ranking table 
total_pass_plays <- plays_df %>%
  count(defensive_team, name = "n_total")
team_pressure <- team_pressure %>%
  left_join(
    total_pass_plays,
    by = "defensive_team"
  ) %>%
  mutate(
    blitz_rate = n / n_total
  )

team_table <- team_pressure %>%
  arrange(desc(nprg)) %>%
  select(
    Team = team_logo_wikipedia,
    `Total Plays` = n,
    `Blitz Rate` = blitz_rate,
    `Average Rushers` = avg_n_rushers,
    `Pressure Rate` = pressure_rate,
    `Exp. Baseline Pressure Rate` = exp_baseline_pressure_rate,
    `Exp. Presure Rate` = exp_pressure_rate,
    NPRG = nprg
  ) %>%
  gt() %>%
  tab_header(
    title = "Team NPRG Rankings",
    subtitle = md("Average NPRG on Pass Blitzes | *2022 Season Weeks 1-9*")
  ) %>%
  gt_img_rows(columns = Team) %>%
  data_color( 
    columns = NPRG, 
    fn = scales::col_numeric( 
      palette = c("red", "white", "green"),
      domain = c(-0.06, 0.06) 
    )
  ) %>%
  fmt_percent(
    -c(Team, `Total Plays`, `Average Rushers`), 
    decimals = 1
  ) %>%
  fmt_number(`Average Rushers`, decimals = 1) %>%
  tab_style(
    style = cell_text(align = "center"),
    locations = cells_body(columns = where(is.numeric))
  ) %>%
  tab_style(
    style = cell_text(align = "center"),
    locations = cells_column_labels(everything()) 
  )
gtsave(team_table, "visualizations/nprg_table.png", expand = 20)