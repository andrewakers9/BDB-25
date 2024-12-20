library(ggplot2)
library(gganimate)
library(ggnewscale)


animate_play <- function(play, 
                         nframes = 100,
                         show_path_ids = NULL) {
  
  if(!is.data.frame(play)) play <- play[[1]]
  
  play <- play %>%
    mutate(
      play_clock = ceiling(40 - frame_id / 10)
    ) 
  
  cols_fill <- c("#FB4F14", "#663300", "#3b3b3b")
  cols_col <- c("#000000", "#663300", "#000000")
  
  if("blitz_prob" %in% colnames(play)) {
    fill_col <- "blitz_prob"
    fill_scale <- scale_fill_gradient2(low = "blue", 
                                       mid = "yellow",
                                       high = "red",
                                       midpoint = 0.5,
                                       na.value = "grey",
                                       name = "Blitz\nProbability",
                                       labels = scales::label_percent())
    play <- play %>%
      mutate(
        blitz_prob = case_when(
          !is.na(blitz_prob) ~ blitz_prob,
          position %in% c("OLB", "DT", "NT", "DE") ~ 1.0,
          TRUE ~ blitz_prob
        )
      )
    play <- play %>%
      group_by(nfl_id) %>%
      fill(blitz_prob, .direction = "down") %>%
      fill(blitz, .direction = "updown")
    blitzers <- play %>%
      filter(
        blitz == 1 | nfl_id %in% show_path_ids,
        frame_id < which(play_df$frame_type == "SNAP")[1] + 50
      ) %>%
      group_by(nfl_id) %>% 
      mutate(
        frame_time = list(1:max(frame_id))
      ) %>%
      ungroup() %>%
      unnest(frame_time) %>%
      filter(frame_time < frame_id) %>%
      mutate(frame_id = as.numeric(frame_time))
    
  } else {
    fill_col <- "club"
    fill_scale <- scale_fill_manual(values = cols_fill, guide = FALSE)
    
  }
  
  teams <- unique(play$club[play$club != "football"])
  play <- play %>%
    mutate(
      club = factor(club, levels = c(teams[1], "football", teams[2]))
    )
  
  # play <- play %>%
  #   mutate(
  #     jersey_number = if_else(club == offense_team, NA, jersey_number)
  #   )
  
  # General field boundaries
  xmin <- 0
  xmax <- 160/3
  hash.right <- 38.35
  hash.left <- 12
  hash.width <- 3.3
  
  #specific boundaries
  ymin <- max(round(min(play$x, na.rm = TRUE) - 20, -1), 0)
  ymax <- min(round(max(play$x, na.rm = TRUE) + 10, -1), 120)
  
  #hash marks
  df.hash <- expand.grid(x = c(0, 23.36667, 29.96667, xmax), y = (10:110))
  df.hash <- df.hash %>% filter(!(floor(y %% 5) == 0))
  df.hash <- df.hash %>% filter(y < ymax, y > ymin)
  
  p1 <- ggplot() +
    
    #setting size and color parameters
    scale_shape_manual(values = c(21, 16, 21), guide = FALSE) +
    scale_size_manual(values = c(6, 4, 6), guide = FALSE) + 
    fill_scale +
    scale_colour_manual(values = cols_col, guide = FALSE) +
    
    #adding hash marks
    #annotate("text", x = df.hash$x[df.hash$x < 55/2], 
    #         y = df.hash$y[df.hash$x < 55/2], label = "_", hjust = 0, vjust = -0.2) + 
    #annotate("text", x = df.hash$x[df.hash$x > 55/2], 
    #         y = df.hash$y[df.hash$x > 55/2], label = "_", hjust = 1, vjust = -0.2) +
    
    #adding yard lines
    annotate("segment", x = xmin, 
             y = seq(max(10, ymin), min(ymax, 110), by = 5), 
             xend =  xmax, 
             yend = seq(max(10, ymin), min(ymax, 110), by = 5),
             color = "white") + 
    
    #adding field yardline text
    annotate("text", x = rep(hash.left, 11), y = seq(10, 110, by = 10), 
             label = c("G   ", seq(10, 50, by = 10), rev(seq(10, 40, by = 10)), "   G"), 
             angle = 270, size = 4, color = "white") + 
    annotate("text", x = rep((xmax - hash.left), 11), y = seq(10, 110, by = 10), 
             label = c("   G", seq(10, 50, by = 10), rev(seq(10, 40, by = 10)), "G   "), 
             angle = 90, size = 4, color = "white") + 
    
    #adding field exterior
    annotate("segment", x = c(xmin, xmin, xmax, xmax), 
             y = c(ymin, ymax, ymax, ymin), 
             xend = c(xmin, xmax, xmax, xmin), 
             yend = c(ymax, ymax, ymin, ymin), colour = "white") + 
    
    #adding players
    geom_point(
      data = play, 
      aes(x = (xmax-y),
          y = x, 
          shape = club,
          fill = !!sym(fill_col),
          group = nfl_id,
          size = club,
          colour = club) 
    ) +  
    
    #adding jersey numbers
    geom_text(data = play, aes(x = (xmax-y), y = x, label = jersey_number), colour = "black", 
              vjust = 0.36, size = 3.5) +
    
    geom_path(
      data = blitzers,
      aes(x = (xmax - y),
          y = x,
          group = nfl_id),
      linewidth = 1.2,
      alpha = 0.6,
      color = "white"
    ) + 
    
    # play clock 
    geom_text(
      data = play %>%
        filter(frame_type != "AFTER_SNAP"),
      aes(
        x = xmax - 10,
        y = ymax - 2.5,
        label = paste0("Play clock: ", play_clock)
      ),
      color = "red",
      size = 5
    )
  
max_frame <- max(play$frame_id)
if(nframes > max_frame) nframes <- max_frame
    
animate(
  
  p1 +
    
    #applying plot limits
    ylim(ymin, ymax) + 
    coord_fixed() +
    
    #theme
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title = element_blank(),
      axis.text = element_blank(),
      panel.background = element_rect(fill = "#85b034"),
      legend.title = element_text(hjust = 0.5)
    ) +
    
    #setting animation parameters
    transition_time(frame_id)  +
    ease_aes('linear') + 
    NULL,
  
  nframes = nframes,
  fps = as.integer(nframes / round(max(play$frame_id)) * 10)
  
)
    
}
#animate_play(play_df, nframes = 200)



