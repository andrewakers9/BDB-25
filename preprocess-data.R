library(future)
library(furrr)
library(tidyverse)

# setwd("C:/Users/andre/OneDrive/Desktop/BDB-25")

# set multi-processing strategy
future::plan(strategy = "cluster", workers = 5)

DEFENSE_POS <- c(
  "CB", "DB", "DE", "DT", "FS", "ILB", "LB", "MLB", "NT", "OLB", "SS"
)

read_tracking_data <- function(week_num, dir = "data/") {
  file_name <- paste0(dir, "tracking_week_", week_num, ".csv")
  df <- read_csv(file_name, col_types = cols())
  return(df)
}

deg_2_rad <- function(deg) {
  rad <- deg * pi /180
  return(rad)
}

get_pass_rushers <- function(play_df, def_team) {
  
  LOS <- play_df %>%
    filter(
      display_name == "football"
    ) %>%
    slice(1) %>%
    select(x, y)
  
  LINE_SET <- play_df %>%
    filter(event == "line_set") %>%
    slice(1) %>%
    pull(frame_id)
  
  if(!is_empty(LINE_SET)) {
    rushers <- play_df %>%
      filter(
        club == def_team,
        frame_id == LINE_SET + 10,
        x - LOS$x < 1.5,
        abs(y - LOS$y) < 7
      )
    
    return(rushers$nfl_id)
    
  }
  
  return(NULL)
  
}

get_blitz_feats <- function(play_df, 
                            def_team, 
                            post_line_set = TRUE) {
  
  # calculate acceleration 
  play_df <- play_df %>%
    arrange(nfl_id, frame_id) %>%
    group_by(nfl_id) %>%
    mutate(
      acc = s - lag(s),
      x_lag = lag(x, 10),
      y_lag = lag(y, 10)
    ) %>%
    ungroup()
  
  LOS <- play_df %>%
    filter(
      frame_id == 1,
      display_name == "football"
    ) %>%
    select(x, y)
  
  if(post_line_set) {
    
    LINE_SET <- play_df %>%
      filter(event == "line_set") %>%
      slice(1) %>%
      pull(frame_id)
    
    if(!is_empty(LINE_SET)) {
      play_df <- play_df %>%
        filter(
          frame_id >= LINE_SET - 10
        )
    }
    
  }
  
  play_df <- play_df %>%
    filter(
      frame_type != "AFTER_SNAP"
    )
  play_df <- play_df %>%
    mutate(
      rel_x = x - LOS$x,
      rel_y = y - LOS$y,
      rel_x_lag = x_lag - LOS$x,
      rel_y_lag = y_lag - LOS$y,
      speed_x = sin(deg_2_rad(dir)) * s,
      speed_y = cos(deg_2_rad(dir)) * s,
      ox = sin(deg_2_rad(o)),
      oy = cos(deg_2_rad(o)),
      tts = max(frame_id) - frame_id
    )
  feat_df <- play_df %>%
    filter(
      display_name != "football",
    ) %>%
    mutate(
      team_sort = if_else(club == def_team, 1, 2)
    ) %>%
    arrange(frame_id, team_sort, nfl_id) %>%
    select(
      frame_id,
      nfl_id,
      tts,
      rel_x,
      rel_y,
      rel_x_lag,
      rel_y_lag,
      speed_x,
      speed_y,
      ox,
      oy,
      acc
    ) 
  
  return(feat_df)
  
}

# read data 
games_df <- read_csv("data/games.csv", col_types = cols()) %>%
  janitor::clean_names()
players_df <- read_csv("data/players.csv", col_types = cols()) %>%
  janitor::clean_names()
plays_df <- read_csv("data/plays.csv", col_types = cols()) %>%
  janitor::clean_names()
player_play_df <- read_csv(
  "data/player_play.csv", 
  col_types = cols(penaltyNames = col_character(), blockedPlayerNFLId3 = col_number())
) %>%
  janitor::clean_names()
tracking_df <- map_df(1:9, read_tracking_data) %>%
  janitor::clean_names()

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

# label encode player positions for embedding 
players_df <- players_df %>%
  left_join(
    players_df %>%
      distinct(position) %>%
      mutate(
        position_id = row_number() - 1
      ),
    by = "position"
  )

# compute extra play variables
plays_df <- plays_df %>%
  mutate(
    yards_to_goal = if_else(
      possession_team == yardline_side,
      100 - yardline_number,
      yardline_number,
      missing = 50
    ),
    game_clock = ms(str_match(game_clock, "\\d+:\\d{2}")),
    time_remaining = (as.numeric(game_clock) / 60) + (4 - quarter) * 15
  )
plays_df <- plays_df %>%
  left_join(
    games_df %>%
      select(game_id, home_team_abbr),
    by = "game_id"
  ) %>%
  mutate(
    def_rel_score = if_else(
      defensive_team == home_team_abbr, 
      pre_snap_home_score - pre_snap_visitor_score,
      pre_snap_visitor_score - pre_snap_home_score
    )
  ) %>%
  arrange(game_id, play_id)

# mirror tracking data, so all plays are in the same direction
tracking_df <- tracking_df %>%
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

# add player positions to tracking data
tracking_df <- tracking_df %>%
  left_join(
    players_df %>%
      select(nfl_id, position),
    by = "nfl_id"
  )

# nest tracking data for each unique game and play IDs
tracking_df <- tracking_df %>%
  nest(.by = c("game_id", "play_id"))

tracking_df <- tracking_df %>%
  arrange(game_id, play_id)

# add play context variables to nested tracking data 
tracking_df <- tracking_df %>%
  left_join(
    plays_df %>%
      select(
        game_id,
        play_id,
        defensive_team
      ),
    by = c("game_id", "play_id")
  )

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
tracking_df <- tracking_df %>%
  semi_join(
    pass_play_ids,
    by = c("game_id", "play_id")
  )

# add position to player_play to determine if pass rusher is a blitzer (non-DL)
player_play_df <- player_play_df %>%
  left_join(
    players_df %>%
      select(nfl_id, position),
    by = "nfl_id"
  )
player_play_df <- player_play_df %>%
  mutate(
    was_initial_pass_rusher = replace_na(was_initial_pass_rusher, 0)
  )

# determine obvious pass rushers based on tracking data 
tracking_df <- tracking_df %>%
  mutate(
    pass_rushers = future_map2(data, defensive_team, get_pass_rushers)
  )

# pass rusher ID via tracking data 
pass_rusher_play <- tracking_df %>%
  select(
    game_id,
    play_id,
    pass_rushers
  ) %>% 
  unnest(pass_rushers) %>%
  rename(nfl_id = pass_rushers) %>%
  mutate(
    is_play_pass_rusher = 1
  )

# determine blitzers 
player_play_df <- player_play_df %>%
  left_join(
    pass_rusher_play,
    by = c("game_id", "play_id", "nfl_id")
  ) %>%
  mutate(
    is_play_pass_rusher = replace_na(is_play_pass_rusher, 0),
    is_blitzer = if_else(
      was_initial_pass_rusher == 1 & 
        !(position %in% c("DT", "NT", "DE")) &
        (is_play_pass_rusher == 0 | position != "OLB"), 
      1, 0
    ),
    y = case_when(
      is_blitzer == 1 ~ 1,
      position %in% c("DT", "NT", "DE") ~ -1,
      is_blitzer == 0 & was_initial_pass_rusher == 1 ~ -1,
      TRUE ~ 0
    ),
    caused_pressure = as.numeric(caused_pressure),
    y_pressure = if_else(was_initial_pass_rusher == 1, caused_pressure, -1)
  )

blitzer_df <- player_play_df %>%
  filter(
    position %in% DEFENSE_POS
  ) %>%
  select(
    game_id,
    play_id,
    nfl_id,
    pass_rusher = was_initial_pass_rusher,
    y_pressure,
    y
  ) %>%
  arrange(
    game_id,
    play_id,
    nfl_id
  )
label_df <- blitzer_df %>%
  group_by(game_id, play_id) %>%
  mutate(
    player_id = row_number() - 1,
    play_pressure = if_else(any(y_pressure == 1, na.rm = TRUE), 1, 0)
  ) %>%
  ungroup() 
write_csv(label_df, "data/labels.csv")

tracking_df <- tracking_df %>%
  mutate(
    blitz_feats = future_map2(data, defensive_team, get_blitz_feats)
  )
blitz_feat_df <- tracking_df %>%
  select(game_id, play_id, blitz_feats) %>%
  unnest(blitz_feats)
future::plan(strategy = "sequential")

# add binary labels (1 if blitzed)
blitz_feat_df <- blitz_feat_df %>%
  left_join(
    label_df,
    by = c("game_id", "play_id", "nfl_id")
  ) %>%
  mutate(
    pass_rusher = replace_na(pass_rusher, 0)
  )

# add player position ids and indices
blitz_feat_df <- blitz_feat_df %>%
  left_join(
    players_df %>%
      select(nfl_id, position_id),
    by = "nfl_id"
  )
blitz_feat_df <- blitz_feat_df %>%
  left_join(
    games_df %>%
      select(game_id, week),
    by = "game_id"
  )

blitz_play_feat_df <- plays_df %>%
  select(
    game_id,
    play_id,
    yards_to_go,
    yards_to_goal,
    down,
    time_remaining,
    def_rel_score,
    pff_man_zone,
  ) %>%
  mutate(
    first_down = if_else(down == 1, 1, 0),
    second_down = if_else(down == 2, 1, 0),
    third_down = if_else(down == 3, 1, 0),
    is_man = if_else(pff_man_zone == "Man", 1, 0)
  ) %>% 
  select(-c(down, pff_man_zone))

write_csv(blitz_feat_df, "data/feats.csv")
write_csv(blitz_play_feat_df, "data/play_feats.csv")

if(file.exists("data/cv_preds.csv")) {
  
  cv_blitz_preds <- read_csv("data/cv_preds.csv")
  cv_blitz_preds <- cv_blitz_preds %>%
    left_join(
      label_df %>%
        select(game_id, play_id, player_id, nfl_id),
      by = c("game_id", "play_id", "player_id")
    )
  cv_blitz_preds <- cv_blitz_preds %>%
    left_join(
      player_play_df %>%
        select(game_id, play_id, nfl_id, position),
      by = c("game_id", "play_id", "nfl_id")
    )
  cv_blitz_preds <- cv_blitz_preds %>%
    group_by(position, y) %>%
    mutate(
      pred_norm = percent_rank(pred)
    ) %>%
    ungroup()
  blitz_feat_df <- blitz_feat_df %>%
    left_join(
      cv_blitz_preds %>%
        select(
          game_id, 
          play_id, 
          player_id, 
          frame_id,
          blitz_prob = pred,
          blitz_prob_norm = pred_norm,
        ),
      by = c("game_id", "play_id", "player_id", "frame_id")
    )
  blitz_feat_df <- blitz_feat_df %>%
    group_by(game_id, play_id, nfl_id) %>%
    fill(
      c(blitz_prob, blitz_prob_norm),
      .direction = "down"
    ) %>%
    ungroup() %>%
    arrange(
      game_id,
      play_id,
      frame_id,
      player_id,
      nfl_id
    )
  write_csv(blitz_feat_df, "data/feats.csv")
  
}