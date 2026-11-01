# INFO 4670 — Week 12, Assignment 4: SMS Spam Detection

The accompanying Python file implements the six-step workflow on the UCI SMS Spam Collection. It was run against all 5,572 records with a stratified 75/25 split (`random_state=42`). The answers below use the resulting metrics.

## Step 1 — Define the problem

**Q1a.** A false positive is slightly more costly for a carrier because it can block an important legitimate message, such as a bank verification code or a family emergency update. A false negative also matters because a phishing or premium-rate scam can reach the subscriber and cause financial harm.

**Q1b.** I use **precision** as the primary metric because I want a message flagged as spam to be very likely to be spam. That reduces the chance of blocking legitimate texts, which is the more costly error in this setting.

**Q1c.** A classifier that always predicts ham gets **86.6% accuracy**, since 4,825 of 5,572 messages are ham. That figure is misleading because it catches none of the 747 spam messages: spam recall and spam F1 are both 0%.

## Step 2 — Explore the data

The dataset contains 4,825 ham messages (86.6%) and 747 spam messages (13.4%). The average message is about 72 characters for ham and 139 characters for spam, so length is informative but cannot classify messages by itself.

## Step 3 — Feature engineering

**Q3a.** `has_caps_word` is 1 when the message contains a word of at least three uppercase letters, such as “WIN” or “FREE”; this can capture promotional emphasis common in older spam. `has_text_cta` is 1 when the message contains “txt,” “text,” or “reply”; these words often signal a response instruction or paid-text offer.

**Q3b.** `has_caps_word` has the largest spam/ham rate gap among the six features: 70.4% of spam messages contain it versus 7.2% of ham, a 63.3 percentage-point gap. A feature that separates the classes well can reduce class uncertainty after a split, so it can provide high information gain. In the fitted tree, `has_caps_word` is the root feature.

**Q3c.** A realistic missed message could say, “Your monthly statement is ready. Sign in to your account at your bank’s usual website to review it.” It may contain no phone number, money or urgency term, all-caps word, text/reply instruction, and may be shorter than 131 characters, so all six hand-built features could be zero even though the message is phishing.

## Step 4 — Split the data

**Q4a.** `stratify=y` keeps the spam/ham proportions close to the original dataset in both training and test sets. Without it, the minority spam class could be underrepresented in one split, making training less useful and evaluation less stable or misleading.

**Q4b.** I would prefer cross-validation when the dataset is small or when the result could depend too much on which examples happened to land in one test split. It measures performance over several train/validation partitions, giving a more stable estimate.

## Step 5 — Train models

The required depth-4 Decision Tree and Bernoulli Naive Bayes were trained. The tree’s test F1 is 0.689; the depth comparison shows its highest test F1 is 0.796 at depths 5, 6, and unlimited, so a future model-selection step could tune depth using cross-validation rather than selecting it from this test set.

## Step 6 — Evaluate

**Q6a.** The confusion matrices use `[[TN, FP], [FN, TP]]` with spam as the positive class.

- **Decision Tree:** TP = 168 spam messages caught; FN = 19 spam messages missed; FP = 133 legitimate messages wrongly flagged.
- **Naive Bayes:** TP = 164 spam messages caught; FN = 23 spam messages missed; FP = 16 legitimate messages wrongly flagged.

**Q6b.** Naive Bayes wins on the chosen primary metric, precision: **0.911 compared with 0.558** for the Decision Tree, a difference of **0.353 (35.3 percentage points)**. Five-fold cross-validation also gives Naive Bayes a higher mean F1 (0.890 vs. 0.838), which supports its overall performance, though that cross-validation metric is F1 rather than precision.

**Q6c. Recommendation memo.** I recommend Naive Bayes for the carrier’s initial spam filter. Its test precision is 91.1%, compared with 55.8% for the Decision Tree, and it wrongly flags 16 legitimate messages instead of 133. Its weakness is that it misses 23 spam messages, four more than the tree. I would improve it by adding word and character n-gram features, then tuning the decision threshold on a validation set to balance missed spam against blocked legitimate texts.

**Q6d.** `has_caps_word` may miss modern spam written in ordinary professional language without all-caps emphasis. `has_text_cta` may miss phishing messages that ask users to follow a website link or open an app rather than text or reply; both features are simple wording cues that attackers can avoid.

## Results summary

| Model | Accuracy | Precision | Recall | F1-score | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.891 | 0.558 | 0.898 | 0.689 | 168 | 19 | 133 | 1,073 |
| Naive Bayes | 0.972 | 0.911 | 0.877 | 0.894 | 164 | 23 | 16 | 1,190 |

Five-fold stratified CV mean F1: Decision Tree **0.838**; Naive Bayes **0.890**.
