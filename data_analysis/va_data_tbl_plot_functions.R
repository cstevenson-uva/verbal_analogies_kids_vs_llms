### FUNCTIONS TO CREATE TABLES AND PLOTS OF DATA

# function to convert all variables in data to correct type
data_convert_var_type <- function(dat) {
  # make vars correct type
  dat <- dat %>%
    mutate(distractor_salience_high_low = factor(distractor_salience_high_low),
           semantic_distance_near_far = factor(semantic_distance_near_far),
           type_relation_jones = factor(type_relation_jones),
           type_verband = factor(type_verband),
           item_number = factor(item_number)) 
  return(dat)
}

# returns table of acc/prop correct
tbl_acc <- function(dat, acc_cols) {
  tbl_allitems_acc <- dat %>%
    summarise_at(vars(acc_cols), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    pivot_longer(cols = acc_cols,
                 names_to = 'solver',
                 values_to = 'proportion_correct') %>%
    mutate(proportion_correct = round(proportion_correct, 2)) %>%
    arrange(proportion_correct)
}

# output latex table of proportion correct / acc data
print_tbl_latex <- function(tbl_dat, output_file) {
  print(xtable(tbl_dat, type = "latex"), 
        file = output_file,
        include.rownames=FALSE)
} 

# plot proportion correct / acc data
plot_acc_bar <- function(tbl_acc_dat) {
  ggplot(data=tbl_acc_dat, 
         aes(x=reorder(solver, proportion_correct), y=proportion_correct, fill=solver)) +
    geom_bar(stat="identity") +
    geom_text(aes(label=proportion_correct), vjust=1.6, color="black", size=3.5) +
    theme_minimal() +
    xlab(NULL) +  ylab('proportion correct') +
    theme(legend.position="none")
}

# get oefenweb relation_types view
get_oefenweb_relation_types_view <- function(dat, result_cols) {
  dat <- dat %>%
    group_by(type_verband) %>%
    # only select relations for which there are at least 5 items
    filter(n() > 5) %>% 
    summarise_at(vars(result_cols), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    pivot_longer(cols = result_cols, 
                 names_to = 'solver',
                 values_to = 'proportion_correct') %>%
    mutate(proportion_correct = round(proportion_correct, 2))  %>%
    mutate(type_verband = recode(type_verband,
                                 'actie-resultaat' = 'action-result',
                                 'classificatie' = 'classification',
                                 'deel - geheel' = 'part-whole',
                                 'probleem - oplossing' = 'problem-solution',
                                 'delen een eigenschap' = 'share characteristic',
                                 'dezelfde categorie' = 'same category',
                                 'gebrek aan' = 'lacks aspect',
                                 'hoort bij elkaar' = 'belong together',
                                 'item - eigenschap' = 'item-function',
                                 'mate/sterkte' = 'degree|strength',
                                 'object - functie' = 'object-function',
                                 'object - locatie' = 'object-location',
                                 'oorzaak-gevolg' = 'cause-effect',
                                 'synoniem' = 'synonymns',
                                 'tegengestelden' = 'opposites',
                                 'teken van' = 'indicates',
                                 'uitvoerder - actie' = 'actor-action')) %>%
    arrange(proportion_correct)
  return(dat)
}

recode_type_relation_jones <- function(dat) {
  dat <- dat %>%
    # group categories together according to Jones 2022 paper
    mutate(type_relation_jones = recode(type_verband,
                                        'actie-resultaat' = 'causal',
                                        'classificatie' = 'categorical',
                                        'deel - geheel' = 'compositional',
                                        'probleem - oplossing' = 'causal',
                                        'delen een eigenschap' = 'compositional',
                                        'dezelfde categorie' = 'categorical',
                                        'gebrek aan' = 'NA',
                                        'hoort bij elkaar' = 'NA',
                                        'item - eigenschap' = 'compositional',
                                        'mate/sterkte' = 'NA',
                                        'object - functie' = 'compositional',
                                        'object - locatie' = 'NA',
                                        'oorzaak-gevolg' = 'causal',
                                        'synoniem' = 'NA',
                                        'tegengestelden' = 'NA',
                                        'teken van' = 'NA',
                                        'uitvoerder - actie' = 'NA',
                                        'grammatica' = 'NA')) 
  return(dat)
}

# get jones relation_types view
get_jones_relation_types_view <- function(dat, result_cols) {
  dat <- dat %>%
    filter(type_relation_jones == 'categorical' | 
             type_relation_jones == 'causal' |
             type_relation_jones == 'compositional') %>%
    group_by(type_relation_jones) %>%
    # examine groups containing at least 10 items
    filter(n() > 10) %>% 
    summarise_at(vars(result_cols), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    pivot_longer(cols = result_cols,
                 names_to = 'solver',
                 values_to = 'proportion_correct') %>%
    mutate(proportion_correct = round(proportion_correct, 2)) %>%
    arrange(proportion_correct)
  return(dat)
}

# get high/low distractor salience view
get_dist_sal_view <- function(dat, result_cols) {
  dat <- dat %>%
    group_by(distractor_salience_high_low) %>%
    summarise_at(vars(result_cols), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    pivot_longer(cols = result_cols, 
                 names_to = 'solver',
                 values_to = 'proportion_correct') %>%
    mutate(proportion_correct = round(proportion_correct, 2)) %>%
    arrange(proportion_correct)
  return(dat)
}

# get near/far view
get_near_far_view <- function(dat, result_cols) {
  dat <- dat %>%
    group_by(semantic_distance_near_far) %>%
    summarise_at(vars(result_cols), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    pivot_longer(cols = result_cols, 
                 names_to = 'solver',
                 values_to = 'proportion_correct') %>%
    mutate(proportion_correct = round(proportion_correct, 2)) %>%
    arrange(proportion_correct)
  return(dat)
}

# plot lines comparing type_verband
plot_acc_line_verband <- function(dat) {
  ggplot(dat, 
         aes(x=reorder(type_verband, proportion_correct), y=proportion_correct, group=solver)) +
    geom_line(aes(color=solver)) +
    geom_point(aes(color=solver)) +
    scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 10)) +
    theme(legend.position="right") +
    xlab(NULL) +  ylab('proportion correct') 
}

# plot lines comparing relation type jones
plot_acc_line_relation_jones <- function(dat) {
  ggplot(dat, 
         aes(x=reorder(type_relation_jones, proportion_correct), y=proportion_correct, group=solver)) +
    geom_line(aes(color=solver)) +
    geom_point(aes(color=solver)) +
    theme(legend.position="right") +
    xlab(NULL) +  ylab('proportion correct') 
}

# plot bar chart comparing relation type jones
plot_acc_bar_relation_jones <- function(dat) {
  ggplot(dat, 
         aes(x=reorder(type_relation_jones, proportion_correct), y=proportion_correct, group=solver)) +
    geom_line(aes(color=solver)) +
    geom_point(aes(color=solver)) +
    theme(legend.position="right") +
    xlab(NULL) +  ylab('proportion correct') 
  # bar chart comparing kids, adults, gpt-3 and gpt-4 on sum correct
  dat %>% 
    group_by(participant_group, alphabet) %>%
    summarise(N = length(sum_correct),
              mean_sum_correct = mean(sum_correct),
              sd_sum_correct = sd(sum_correct),
              se_sum_correct = sd(sum_correct)/sqrt(length(sum_correct))) %>%
    ggplot(aes(x = alphabet, y = mean_sum_correct, color = participant_group, fill=participant_group)) + 
    geom_bar(stat = "identity", position = position_dodge()) +
    geom_errorbar(aes(ymin = mean_sum_correct - se_sum_correct, 
                      ymax = mean_sum_correct + se_sum_correct),
                  width = .2,
                  position = position_dodge(.9)) +
    labs(x="Alphabet", y = "Total Correct", color="Participant Group", fill="Participant Group")
}

# plot lines comparing near far
plot_acc_near_far <- function(dat) {
  ggplot(dat, 
         aes(x=reorder(semantic_distance_near_far, proportion_correct), y=proportion_correct, group=solver)) +
    geom_line(aes(color=solver)) +
    geom_point(aes(color=solver)) +
    theme(legend.position="right") +
    xlab(NULL) +  ylab('proportion correct') 
}

# plot lines comparing near far
plot_acc_dist_sal <- function(dat) {
  ggplot(dat, 
         aes(x=reorder(distractor_salience_high_low, proportion_correct), y=proportion_correct, group=solver)) +
    geom_line(aes(color=solver)) +
    geom_point(aes(color=solver)) +
    theme(legend.position="right") +
    xlab(NULL) +  ylab('proportion correct') 
}

 