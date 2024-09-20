# Load necessary libraries
library(readr)
library(dplyr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(car)
# Load the dplyr package
library(dplyr)
library(effects)
# Read data
data <- read_csv("/Users/tshoiasc/processed_rapeseed_data.csv")


# Calculate the number of observations for each Branch Level
branch_level_counts <- data %>%
  group_by(Branch_Level) %>%
  summarise(Count = n())

# Output the results
print(branch_level_counts)
# Data preprocessing
data <- data %>%
  filter(view == "0") %>%  # Only keep data with view sv-000
  filter(!is.na(New_Bud_ID), Branch_Level == 2) %>%  # Only keep data with New Bud ID and Branch_Level equal to 2
  mutate(
    Date = as.Date(Date, format = "%Y/%m/%d"),
    Time = as.numeric(Date - min(Date)),  # Convert date to numeric time (days starting from 0)
    Bud_Position_X = as.numeric(gsub("\\(|\\)", "", sapply(strsplit(Bud_Position, ","), `[`, 1))),
    Bud_Position_Y = as.numeric(gsub("\\(|\\)", "", sapply(strsplit(Bud_Position, ","), `[`, 2))),
    vernalization_temp = as.factor(vernalization_temp),
    crop_type = as.factor(crop_type),
    Branch_ID = paste(plant_id, New_Bud_ID, sep = "_"),  # No longer need view since only sv-000 view is kept
    Angle = abs(Angle)  # Take the absolute value of branch angle
  )

# Diagnostic code
print("Number of levels in each factor:")
print(paste("vernalization_temp:", nlevels(data$vernalization_temp)))
print(paste("crop_type:", nlevels(data$crop_type)))
print(paste("Number of unique plants:", length(unique(data$plant_id))))
print(paste("Number of unique branches:", length(unique(data$Branch_ID))))

# Create mixed-effects models
model_length <- lmer(Length ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)
model_angle <- lmer(Angle ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data, 
                    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
model_bud_position_x <- lmer(Bud_Position_X ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)
model_bud_position_y <- lmer(Bud_Position_Y ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)

# Output model summaries
print("Length")
summary(model_length)
print("Angle")
summary(model_angle)
# print("Bud Position X")
# summary(model_bud_position_x)
# print("Bud Position Y")
# summary(model_bud_position_y)


# Extract residuals
residuals_length <- resid(model_length)

# Create QQ plot and add title
qqnorm(residuals_length, main = "QQ Plot of Residuals for Model Length")
qqline(residuals_length, col = "red")
# Kolmogorov-Smirnov test
ks.test(residuals_length, "pnorm", mean = mean(residuals_length), sd = sd(residuals_length))


# Extract residuals for the Angle model
residuals_angle <- resid(model_angle)


# Create QQ plot
qqnorm(residuals_angle)
qqline(residuals_angle, col = "red")

# Shapiro-Wilk normality test
ks.test(residuals_angle, "pnorm", mean = mean(residuals_angle), sd = sd(residuals_angle))

data$Log_Angle <- log1p(data$Angle)  # Use log1p to avoid log(0) issues
model_log_angle <- lmer(Log_Angle ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)
# Extract residuals for the Log_Angle model
residuals_log_angle <- resid(model_log_angle)

# Create QQ plot
qqnorm(residuals_log_angle, main = "QQ Plot of Residuals for Model Log(Angle)")
qqline(residuals_log_angle, col = "red")  # Add normality reference line

ks.test(residuals_log_angle, "pnorm", mean = mean(residuals_log_angle), sd = sd(residuals_log_angle))

data$Sqrt_Angle <- sqrt(data$Angle)
model_sqrt_angle <- lmer(Sqrt_Angle ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)
residuals_sqrt_angle <- resid(model_sqrt_angle)
qqnorm(residuals_sqrt_angle, main = "QQ Plot of Residuals for Model Sqrt(Angle)")
qqline(residuals_sqrt_angle, col = "red")
summary(model_sqrt_angle)
ks.test(residuals_sqrt_angle, "pnorm", mean = mean(residuals_sqrt_angle), sd = sd(residuals_sqrt_angle))


library(MASS)
data$angle_asinh <- asinh(data$angle)

third = lmer(angle_cube_root ~ Time * vernalization_temp * crop_type + (1|plant_id/Branch_ID), data = data)
residuals_third_angle <- resid(third)
qqnorm(residuals_third_angle)
qqline(residuals_third_angle, col = "red")


min_angle <- min(data$Angle)
if (min_angle <= 0) {
  data$Angle_shifted <- data$Angle - min_angle + 1  # Add a constant to make all values positive
} else {
  data$Angle_shifted <- data$Angle
}

# Model using shifted data
model_glmm <- glmer(Angle_shifted ~ Time * vernalization_temp * crop_type + 
                      (1|plant_id/Branch_ID), 
                    data = data, 
                    family = Gamma(link = "log"))
summary(model_glmm)

# Extract residuals for the Angle model
residuals_angle <- resid(model_glmm)

# Create QQ plot
qqnorm(residuals_angle)
qqline(residuals_angle, col = "red")

anova(model_glmm, type = 3)


# Calculate confidence intervals for the time effect
confint(model_glmm, parm="Time", method="Wald")


# Use likelihood ratio test to compare models with and without time effect
model_without_time <- update(model_glmm, . ~ . - Time)
anova(model_glmm, model_without_time)

# Check the marginal effect of time
plot(effect("Time", model_glmm))
# ANOVA analysis
print("ANOVA for Length")
anova(model_length)
print("ANOVA for Angle")
anova(model_angle)
print("ANOVA for Bud Position X")
anova(model_bud_position_x)
print("ANOVA for Bud Position Y")
anova(model_bud_position_y)

# Visualization
# Custom color palette
color_palette <- c('#4C72B0', '#55A868', '#C44E52', '#8172B3', 
                   '#CCB974', '#64B5CD', '#6A4C93', '#937860', 
                   '#DA8BC3', '#8C8C8C')

# Visualize the interaction effect of Time * vernalization_temp * crop_type with more prominent fit lines and custom colors
ggplot(data, aes(x = Time, y = Length, color = crop_type)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +  # Use default color mapping
  facet_grid(vernalization_temp ~ crop_type) +
  scale_color_manual(values = color_palette) +  # Apply custom color palette
  theme_minimal() +
  labs(title = "Length over Time by Vernalization Temperature and Crop Type",
       x = "Time (days)", y = "Length")


# Save models
saveRDS(model_length, "model_length.rds")
saveRDS(model_angle, "model_angle.rds")
saveRDS(model_bud_position_x, "model_bud_position_x.rds")
saveRDS(model_bud_position_y, "model_bud_position_y.rds")


# Assuming your data is stored in the variable 'data'
library(fitdistrplus)

min_angle <- min(data$Log_Angle)
if (min_angle <= 0) {
  data$Log_Angle_shifted <- data$Log_Angle - min_angle + 1  # Add a constant to make all values positive
} else {
  data$Log_Angle_shifted <- data$Log_Angle
}
# Fit Gamma distribution
fit_gamma <- fitdist(data$Log_Angle_shifted, "gamma")

# Perform Kolmogorov-Smirnov test
ks_test <- ks.test(data$Log_Angle_shifted, "pgamma", shape = fit_gamma$estimate["shape"], 
                   rate = fit_gamma$estimate["rate"])

print(ks_test)

# Create Q-Q plot
plot(fit_gamma)
