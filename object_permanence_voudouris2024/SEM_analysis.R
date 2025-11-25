## Run Logistic Regression & SEM on data

# setwd("<measurement-layouts-repo>")

library(tidyverse)
library(lavaan)

data <- read.csv("object_permanence_voudouris2024/results_final_clean_long.csv")

brier_score <- function(pred_probs, actual_outcomes) {
  if (length(pred_probs) != length(actual_outcomes)) {
    stop("Predicted probabilities and actual outcomes must have the same length.")
  }
  
  squared_diff <- (pred_probs - actual_outcomes)^2
  
  bs <- mean(squared_diff)
  return(bs)
}


agents <- c("dreamer-bc-all",
            "dreamer-bc_opc-all",
            "dreamer-bc_opc-strat",
            "dreamer-bc_opc_opt-all",
            "dreamer-bc_opc_opt-strat",
            "ppo-bc-all_2023",
            "ppo-bc_opc-strat_2023",
            "ppo-bc_opc-all_2023",
            "ppo-bc_opc_opt-strat_2023",
            "ppo-bc_opc_opt-all_2023",
            "Random Action Agent no bias no correlation uniform step length max 20_2023",
            "Vanilla Braitenberg 15 rays over 60 degs_2023",
            "Child")

agent_names_short <- c(
  "Dreamer 1",
  "Dreamer 2",
  "Dreamer 3",
  "Dreamer 4",
  "Dreamer 5",
  "PPO 1",
  "PPO 2",
  "PPO 3",
  "PPO 4",
  "PPO 5",
  "Random Action Agent",
  "Heuristic Agent",
  "Child"
)


logistic_formula_success <- as.formula("success ~ minDistToGoal + minDistToCorrectChoice + minNumTurnsGoal + minNumTurnsChoice + goalMaxDistEuclidean + mainGoalSize + numChoices + lavaPresence + goalRightRelToStart + goalCentreRelToStart + goalLeftRelToStart + goalBecomesAllocentricallyOccluded")

logistic_formula_choice <- as.formula("correctChoice ~ minDistToGoal + minDistToCorrectChoice + minNumTurnsGoal + minNumTurnsChoice + goalMaxDistEuclidean + mainGoalSize + numChoices + lavaPresence + goalRightRelToStart + goalCentreRelToStart + goalLeftRelToStart + goalBecomesAllocentricallyOccluded")

logistic_briers_success <- c()
logistic_briers_choice <- c()


for (agent in agents){
  set.seed(1997)
  data_subset <- data %>% filter((agent_tag_seed == agent | agent_type_gen == agent))
  
  train_index <- sample(nrow(data_subset), 0.8 * nrow(data_subset))
  train_data <- data_subset[train_index, ]
  test_data <- data_subset[-train_index, ]
  
  cat("\n\nRunning regressions for agent:", agent, "\n\n")
  cat("Training set size:", nrow(train_data), "\n")
  cat("Testing set size:", nrow(test_data), "\n\n")
  
  log_reg_model_success <- glm(logistic_formula_success,
                               data = train_data,
                               family = binomial(link = "logit"))
  
  log_reg_probs_success <- predict(log_reg_model_success, newdata = test_data, type = "response")
  bs_log_reg_success <- brier_score(log_reg_probs_success, test_data$success)
  cat("\n--- DV 1 Prediction Evaluation ---\n")
  cat("Brier Score (Success):", round(bs_log_reg_success, 4), "\n")
  logistic_briers_success <- c(logistic_briers_success, bs_log_reg_success)
  
  log_reg_model_choice <- glm(logistic_formula_choice,
                               data = train_data,
                               family = binomial(link = "logit"))
  
  log_reg_probs_choice <- predict(log_reg_model_choice, newdata = test_data, type = "response")
  bs_log_reg_choice <- brier_score(log_reg_probs_choice, test_data$correctChoice)
  cat("\n--- DV 2 Prediction Evaluation ---\n")
  cat("Brier Score (Choice):", round(bs_log_reg_choice, 4), "\n")
  logistic_briers_choice <- c(logistic_briers_choice, bs_log_reg_choice)
  
  
}

sem_model_both <- '
  
  navigationSuccess =~ minDistToGoal + minNumTurnsGoal
  navigationChoice =~ minDistToCorrectChoice + minNumTurnsChoice
  objectPermanenceSuccess =~ numChoices + minDistToGoal + goalBecomesAllocentricallyOccluded
  objectPermanenceChoice =~ numChoices + minDistToCorrectChoice + goalBecomesAllocentricallyOccluded
  visualAcuity =~ goalMaxDistEuclidean + mainGoalSize
  lavaAbility =~ lavaPresence
  rightAbility =~ goalRightRelToStart
  leftAbility =~ goalLeftRelToStart
  centreAbility =~ goalCentreRelToStart
  
  # Regressions on both DVs
  success ~ navigationSuccess + objectPermanenceSuccess + lavaAbility
  correctChoice ~ navigationChoice + objectPermanenceChoice + visualAcuity + rightAbility + leftAbility + centreAbility
  
  # Allow DVs to correlate
  success ~~ correctChoice
'

sem_both_briers_success <- c()
sem_both_briers_choice <- c()

for (agent in agents){
  set.seed(1997)
  data_subset <- data %>% filter((agent_tag_seed == agent | agent_type_gen == agent))
  
  train_index <- sample(nrow(data_subset), 0.8 * nrow(data_subset))
  train_data <- data_subset[train_index, ]
  test_data <- data_subset[-train_index, ]
  
  sem_fit_both <- sem(sem_model_both, data = train_data,
                      estimator = "ML", missing = "fiml")
  
  sem_pred_both <- lavPredictY(sem_fit_both, newdata = test_data, xnames = c("minDistToGoal", 
                                                                             "minDistToCorrectChoice", 
                                                                             "minNumTurnsGoal",
                                                                             "minNumTurnsChoice",
                                                                             "goalMaxDistEuclidean",
                                                                             "mainGoalSize",
                                                                             "numChoices",
                                                                             "lavaPresence",
                                                                             "goalRightRelToStart",
                                                                             "goalCentreRelToStart",
                                                                             "goalLeftRelToStart",
                                                                             "goalBecomesAllocentricallyOccluded"),
                               ynames = c("success", "correctChoice"))
  
  sem_probs_both_success <- pmin(pmax(sem_pred_both[, "success"], 0), 1)
  sem_probs_both_choice <- pmin(pmax(sem_pred_both[, "correctChoice"], 0), 1)
  
  bs_sem_both_success <- brier_score(sem_probs_both_success, test_data$success)
  bs_sem_both_choice <- brier_score(sem_probs_both_choice, test_data$correctChoice)
  
  cat("Brier Score - Success (SEM Both):", round(bs_sem_both_success, 4), "\n")
  cat("Brier Score - Choice (SEM Both):", round(bs_sem_both_choice, 4), "\n\n")
  
  sem_both_briers_success <- c(sem_both_briers_success, bs_sem_both_success)
  sem_both_briers_choice <- c(sem_both_briers_choice, bs_sem_both_choice)
  
}

final_results <- data.frame(
  `Name Short` = agent_names_short,
  `Agent Name` = agents,
  `Logistic Brier Success` = logistic_briers_success,
  `Logistic Brier Choice` = logistic_briers_choice,
  `SEM (Both Dependents) Brier Success` = sem_both_briers_success,
  `SEM (Both Dependents) Brier Choice` = sem_both_briers_choice
)

write.csv(final_results, "object_permanence_voudouris2024/logistic_sem_briers.csv", row.names = FALSE)
