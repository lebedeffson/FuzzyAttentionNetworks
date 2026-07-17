# Methods for paper

We used the 4,000-patient PhysioNet/CinC 2012 Set A cohort with a frozen patient-level train/validation/calibration/test split of 2400/600/400/600. Models received the first 48 hours represented as values, observation masks, and elapsed-time channels. All normalization and proxy-concept thresholds were estimated on training data only. Five canonical architectures were evaluated across 30 crossed initialization/data-order seeds.
