source("~/Settings/startup.R")

input_filename = "/Users/erindb/Projects/DisExtract/model/exp/confusion_test.csv"

filename = input_filename

df = read.csv(filename) %>%
  mutate(labels = num(labels),
         preds = num(preds))
print(paste("overall:", mean(df$labels==df$preds)))
print(paste("mean:",
            with(df %>% group_by(labels) %>% 
                   summarise(accuracy = mean(labels==preds)),
                 mean(accuracy))))

class_labels = c(
    "after", "also", "although", "and", "as",
    "because", "before", "but", #"for example",
    #"however", 
    "if", "so", "still",
    "then", "though", "when", "while")

df = df %>%
  mutate(labels = factor(labels, levels=0:(length(class_labels)-1), labels=class_labels),
         preds = factor(preds, levels=0:(length(class_labels)-1), labels=class_labels))


confusions = table(df) %>% as.data.frame() %>%
  mutate(
    labels = char(labels),
    preds = char(preds)
  ) %>%
  group_by(labels) %>%
  mutate(prop_gold = Freq / sum(Freq)) %>%
  ungroup() %>%
  group_by(preds) %>%
  mutate(prop_classifications = Freq / sum(Freq)) %>%
  ungroup()
diagonal = confusions %>% filter(labels==preds)

clustering = "gold"

class_freqs = df %>% group_by(labels) %>%
  summarise(freq=length(labels))
freq_levels = class_freqs$labels[order(class_freqs$freq)]

confusions = confusions %>%
  mutate(labels = factor(labels, levels=freq_levels),
         preds = factor(preds, levels=freq_levels))
  
p = confusions %>%
  mutate(Confusion = log(prop_gold+0.001)) %>%
  ggplot(., aes(x=preds, y=labels, fill=Confusion)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  xlab("Model Classification")
print(p)

ggsave("confusion.png", width=6, height=4)