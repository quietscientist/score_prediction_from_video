Ensure PYTHONUNBUFFERED=1 before commands that include logging to avoid python buffering stdout when piped to tee

never access /home/msegado/tapedeck/msegado/PANDA without explicit permission from the user.

## Dataset Paths
- GMA pose JSONs: /home/msegado/tapedeck/msegado/PANDA/outputs/PANDA2B/CHOP_body_kp
- GMA scores CSV: lives inside the CHOP_body_kp dir as gma_scores.csv (or passed via --scores-file)
- UDysRS: /home/msegado/tapedeck/msegado/Datasets/linearprobe/UDysRS_UPDRS_Export/