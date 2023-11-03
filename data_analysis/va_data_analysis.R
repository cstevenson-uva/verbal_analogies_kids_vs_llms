library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(xtable)

### IMPORT DATA
# set dir
setwd('~/Research/verbalAnalogies/OefenwebVerbalAnalogies')
# get data for RQ1 & RQ2
dat_RQ1_2 <- read.csv('data/va_kids_vs_llms_ABC_results.csv')
# get data for RQ3, Experiments 1-3
dat_EXP1_3 <- read.csv('data/va_kids_vs_xlmv_gpt3_results.csv')
# get data for RQ3, Experiment 4
dat_EXP4 <- read.csv('data/va_kids_vs_xlmv_gpt3_ACB_new_results.csv')
# get table of relation types used in va dataset
reltypes_overview <- read.csv('data/data_item_factors/relation_types.csv')

# load functions for producing tables and plots
source('data_analysis/va_data_tbl_plot_functions.R')

#### RQ1: How well do LLMs perform compared to children ages 7-12 in verbal analogy solving?
# prep RQ1 data
dat_RQ1_2 <- data_convert_var_type(recode_type_relation_jones(dat_RQ1_2))
  
# make lists of cols that hold accuracy results for A:B::C:? items
acc_cols_all_models <- c('result_robbert', 'result_bertje', 'result_xlmv', 'result_gpt2',
                         'result_mgpt', 'result_gpt3', 'result_w2v', 'result_ft')
acc_cols_best_models <- c('result_xlmv', 'result_gpt3')
acc_cols_all_ages <- c('prob_correct_7', 'prob_correct_8', 'prob_correct_9', 
                       'prob_correct_10', 'prob_correct_11', 'prob_correct_12')
acc_cols_kids_avg <- c('prob_correct_children')

# create table and plot showing kids vs LLM accuracy on all items with A:B::C:? prompt
#tbl_acc_kids_vs_llms <- tbl_acc(va_dat, c(acc_cols_all_models, acc_cols_all_ages, acc_cols_kids_avg))
tbl_acc_kids_vs_llms <- tbl_acc(dat_RQ1_2, c(acc_cols_all_models, acc_cols_all_ages, acc_cols_kids_avg))
#print_tbl_latex(tbl_acc_kids_vs_llms, "tables_figures/tbl_acc_children_LLMs_all_items.tex")
plot_acc_bar(tbl_acc_kids_vs_llms)
#dat_RQ1_2 = va_dat
# print num items accuracy is based on
sum(!is.na(dat_RQ1_2$result_w2v))
sum(!is.na(dat_RQ1_2$result_ft))
sum(!is.na(dat_RQ1_2$result_mgpt))
sum(!is.na(dat_RQ1_2$result_gpt2))
sum(!is.na(dat_RQ1_2$result_bertje))
sum(!is.na(dat_RQ1_2$result_robbert))
sum(!is.na(dat_RQ1_2$result_gpt3))
sum(!is.na(dat_RQ1_2$result_xlm))

#### RQ2: Which item factors influence analogy solving? (for kids vs xlmv vs gpt3)
# transform data from wide to long
dat_RQ2_long <- dat_RQ1_2 %>%
  # drop cols we don't need
  select(-c(type_verband:rating_uncertainty)) %>%
  select(-c(result_w2v:result_robbert)) %>%
  # make longer with correct (0-1) as DV
  pivot_longer(cols = c(acc_cols_best_models, acc_cols_kids_avg, acc_cols_all_ages),
               names_to = 'solver',
               values_to = 'correct') %>%
  mutate(correct = round(correct, 5)) %>%
  # make sure solver is correct type
  mutate(solver = factor(solver)) %>%
  # filter solver avg children, xlmv and gpts
  filter(solver == 'prob_correct_children' |
           solver == 'result_xlmv' |
           solver == 'result_gpt3') %>%
  # rename the solvers for plots
  mutate(solver = ifelse(solver == 'prob_correct_children', 'children', 
                         ifelse(solver == 'result_xlmv', 'XLM-V', 'GPT-3')))

### RELATION TYPE JONES
# data view best models, avg kids
tbl_jones_reltype_acc_best <- get_jones_relation_types_view(dat_RQ1_2, c(acc_cols_best_models, acc_cols_kids_avg))
plot_acc_line_relation_jones(tbl_jones_reltype_acc_best)
# data view kids all ages
tbl_jones_reltype_acc_kids <- get_jones_relation_types_view(dat_RQ1_2, acc_cols_all_ages)
plot_acc_line_relation_jones(tbl_jones_reltype_acc_kids)
# get rel type view from long
reltype_jones_dat <- dat_RQ2_long %>%
  filter(type_relation_jones == 'causal' | 
           type_relation_jones == 'compositional' |
           type_relation_jones == 'categorical') # note: 302 items dropped
# bar plot of differences in relation type between kids and best llms
reltype_jones_dat %>%
  group_by(solver, type_relation_jones) %>%
  summarise(N = length(correct),
            mean_correct = mean(correct),
            sd_correct = sd(correct),
            se_correct = sd(correct)/sqrt(length(correct))) %>%
  ggplot(aes(x = type_relation_jones, y = mean_correct, color = solver, fill=solver)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                width = .2, color = "black",
                position = position_dodge(.9)) +
  labs(x="Relation Type", y = "Accuracy", color="Solver", fill="Solver")

## mixed logistic regression model
# relevel factor reltype jones to easily test hypotheses
reltype_jones_dat$type_relation_jones <- relevel(reltype_jones_dat$type_relation_jones, "compositional")
reltype_jones_dat_xlmv <- reltype_jones_dat %>% 
  filter(solver == 'XLM-V')
reltype_jones_dat_gpt3 <- reltype_jones_dat %>% 
  filter(solver == 'GPT-3')
reltype_jones_dat_children <- reltype_jones_dat %>% 
  filter(solver == 'children')

reltype_jones_glm_xlmv <-glm(correct ~ type_relation_jones,
                             family = binomial,
                             data = reltype_jones_dat_xlmv)
reltype_jones_glm_gpt3 <- glm(correct ~ type_relation_jones,
                              family = binomial,
                              data = reltype_jones_dat_gpt3)
reltype_jones_glm_children <- glm(correct ~ type_relation_jones,
                                  family = binomial,
                                  data = reltype_jones_dat_children)
summary(reltype_jones_glm_xlmv) 
summary(reltype_jones_glm_gpt3) 
summary(reltype_jones_glm_children) 

reltype_jones_dat_causal <- reltype_jones_dat %>% 
  filter(type_relation_jones == 'causal')
reltype_jones_dat_causal$solver <- relevel(reltype_jones_dat_causal$solver, "XLM-V")

reltype_jones_glm_causal <- glm(correct ~ solver,
                             family = binomial,
                             data = reltype_jones_dat_causal)
summary(reltype_jones_glm_causal) 

reltype_jones_glm_fit <- glm(correct ~ solver * type_relation_jones,
                             family = binomial,
                             data = reltype_jones_dat)
summary(reltype_jones_glm_fit) 

### NEAR VS FAR
# data view all models
tbl_acc_near_far <- get_near_far_view(dat_RQ1_2, c(acc_cols_all_models, acc_cols_kids_avg))
plot_acc_near_far(tbl_acc_near_far)
# data view kids all ages
tbl_acc_near_far_kids <- get_near_far_view(dat_RQ1_2, acc_cols_all_ages)
plot_acc_near_far(tbl_acc_near_far_kids)
# get nearfar view from long
nearfar_dat <- dat_RQ2_long %>%
  filter(semantic_distance_near_far == 'near' | 
           semantic_distance_near_far == 'far') # note 217 items removed
# bar plot of differences in relation type between kids and best llms
nearfar_dat %>%
  group_by(solver, semantic_distance_near_far) %>%
  summarise(N = length(correct),
            mean_correct = mean(correct, na.rm = TRUE),
            sd_correct = sd(correct, na.rm = TRUE),
            se_correct = sd(correct, na.rm = TRUE)/sqrt(length(correct))) %>%
  ggplot(aes(x = semantic_distance_near_far, y = mean_correct, color = solver, fill=solver)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                width = .2, color = "black",
                position = position_dodge(.9)) +
  labs(x="Semantic Distance Base to Target", y = "Accuracy", color="Solver", fill="Solver")
## mixed logistic regression model
# change kids probability to 1/0 for analysis
#reltype_jones_dat$correct <- ifelse(reltype_jones_dat$correct >= .5, 1, 0) 
nearfar_glm_fit <- glm(correct ~ semantic_distance_near_far * solver,
                             family = binomial,
                             data = nearfar_dat)
summary(nearfar_glm_fit)

### DISTRACTOR SALIENCE
# data view all models
tbl_acc_dist_sal <- get_dist_sal_view(dat_RQ1_2, c(acc_cols_all_models, acc_cols_kids_avg))
plot_acc_dist_sal(tbl_acc_dist_sal)
# data view kids all ages
tbl_acc_dist_sal_kids <- get_dist_sal_view(dat_RQ1_2, acc_cols_all_ages)
plot_acc_dist_sal(tbl_acc_dist_sal_kids)
# get distractor salience view from long
distractor_dat <- dat_RQ2_long %>%
  filter(!is.na(distractor_salience_high_low))
# bar plot of differences between kids and best llms
distractor_dat %>%
  group_by(solver, distractor_salience_high_low) %>%
  summarise(N = length(correct),
            mean_correct = mean(correct, na.rm = TRUE),
            sd_correct = sd(correct, na.rm = TRUE),
            se_correct = sd(correct, na.rm = TRUE)/sqrt(length(correct))) %>%
  ggplot(aes(x = distractor_salience_high_low, y = mean_correct, color = solver, fill=solver)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_errorbar(aes(ymin = mean_correct - se_correct, 
                    ymax = mean_correct + se_correct),
                width = .2, color = "black",
                position = position_dodge(.9)) +
  labs(x="Distractor Salience", y = "Accuracy", color="Solver", fill="Solver")
## mixed logistic regression model
distractor_glm_fit <- glm(correct ~ distractor_salience_high_low * solver,
                             family = binomial,
                             data = distractor_dat)
summary(distractor_glm_fit) 
## RQ1&2 OTHER PLOTS/TABLES 
# TBL RELATION TYPE
tbl_reltypes_overview <- reltypes_overview %>%
  select(-c(type_verband, A, B, C, D, distractor_1, distractor_2, distractor_3, distractor_4))
## by oefenweb relation type
tbl_reltype_acc <- get_oefenweb_relation_types_view(dat_RQ1_2, c(acc_cols_all_models, acc_cols_kids_avg))
plot_acc_line_verband(tbl_reltype_acc)
# data view kids all ages
tbl_reltype_acc_kids <- get_oefenweb_relation_types_view(dat_RQ1_2, acc_cols_all_ages)
plot_acc_line_verband(tbl_reltype_acc_kids)

### RQ3: EXPERIMENT 1 C prompt
# get C: results
C_results <- data_convert_var_type(recode_type_relation_jones(dat_EXP1_3))
# cols that hold acc info for C:? solved items: solution by association
C_cols_best_models <- c('result_C_xlmv', 'result_C_gpt3')
ABC_cols_best_models <- c('result_ABC_xlmv', 'result_ABC_gpt3')
## tbls of acc on C
tbl_acc_c_prompt <- tbl_acc(C_results, C_cols_best_models)
## jones_rel_type
tbl_acc_c_prompt_jones <- get_jones_relation_types_view(C_results, c(C_cols_best_models, ABC_cols_best_models)) %>%
                          arrange(type_relation_jones)
## near/far 
tbl_acc_c_prompt_nearfar <- get_near_far_view(C_results, C_cols_best_models)
## high_low DS
tbl_acc_c_prompt_ds <- get_dist_sal_view(C_results, C_cols_best_models)
## what causes acc on C in terms of item factors?
glm_C_xlmv_reljones <- glm(result_C_xlmv ~ type_relation_jones,
                  data = C_results, family=binomial())
summary(glm_C_xlmv_reljones)
glm_C_xlmv_nearfar <- glm(result_C_xlmv ~ semantic_distance_near_far,
                           data = C_results, family=binomial())
summary(glm_C_xlmv_nearfar)
glm_C_xlmv_ds <- glm(result_C_xlmv ~ distractor_salience_high_low,
                          data = C_results, family=binomial())
summary(glm_C_xlmv_ds)
glm_C_gpt3_ds <- glm(result_C_gpt3 ~ distractor_salience_high_low,
                     data = C_results, family=binomial())
summary(glm_C_gpt3_ds)

### EXP 2
# now filter out the items solved by C only
dat_no_C_xlmv <- C_results %>%
  filter(result_C_xlmv == 0)
dat_no_C_gpt3 <- C_results %>%
  filter(result_C_gpt3 == 0)
dat_no_C <- C_results %>%
  filter(result_C_xlmv == 0 & result_C_gpt3 == 0)
# number items not solved by c only for either model
length(dat_no_C_xlmv$item_number)
length(dat_no_C_gpt3$item_number)
length(dat_no_C$item_number)
## tbls of acc
tbl_acc_no_c_xlmv <- tbl_acc(dat_no_C_xlmv, c('result_ABC_xlmv',
                                    acc_cols_all_ages, acc_cols_kids_avg))
tbl_acc_no_c_gpt3 <- tbl_acc(dat_no_C_gpt3, c('result_ABC_gpt3',
                                              acc_cols_all_ages, acc_cols_kids_avg))
tbl_acc_no_c <- tbl_acc(dat_no_C, c('result_ABC_xlmv', 'result_ABC_gpt3',
                                    acc_cols_all_ages, acc_cols_kids_avg))

### EXPERIMENT 3 ACB prompt
ACB_results <- C_results
ACB_cols_best_models <- c('result_ACB_xlmv', 'result_ACB_gpt3')
## tbls of acc
tbl_acc_ACB <- tbl_acc(ACB_results, c(ACB_cols_best_models, acc_cols_all_ages, acc_cols_kids_avg))

### EXPERIMENT 4 ACB new prompt
# dataset
ACB_new_results <- dat_EXP4
ACB_new_cols_best_models <- c('result_xlmv', 'result_gpt3')
## tbls of acc
tbl_acc_ACB_new <- tbl_acc(ACB_new_results, ACB_new_cols_best_models)

### OLDER STUFF
## get acc per model per experiment
mean(va_dat_ACB$result_C_xlmv, na.rm = TRUE) #.28
mean(va_dat_ACB$result_C_gpt3, na.rm = TRUE) #.40
mean(va_dat_ACB$result_C_gpt4, na.rm = TRUE) #.55
mean(va_dat_ACB$result_ACB_xlmv, na.rm = TRUE) #.48
mean(va_dat_ACB$result_ACB_gpt3, na.rm = TRUE) #.48
mean(va_dat_ACB$result_ACB_gpt4, na.rm = TRUE) #.71
