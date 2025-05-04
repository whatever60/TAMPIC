aws s3 sync s3://yiming-qu/data/camii/early_plates_mono ~/data/camii/early_plates_mono --delete --exclude "*" --include "rgb/*" --include "*.json"
aws s3 sync s3://yiming-qu/data/camii/early_plates_dual ~/data/camii/early_plates_dual --delete --exclude "*" --include "rgb/*" --include "*.json"
aws s3 sync s3://yiming-qu/data/camii/op_plates ~/data/camii/op_plates --delete --exclude "*" --include "rgb/*" --include "*.json"
aws s3 sync s3://yiming-qu/data/camii/op_plates_combined ~/data/camii/op_plates_combined --delete --exclude "*" --include "rgb/*" --include "*.json"
aws s3 sync s3://yiming-qu/data/camii ~/data/camii --exclude "*/" --include "*.json"
aws s3 sync s3://yiming-qu/data/camii ~/data/camii --exclude "*" --include "*_pc3.png"

# For training without hyperspectral data
aws s3 sync s3://yiming-qu/data/camii ~/data/camii --exclude "_*" --exclude "*.fastq.gz" --exclude "*.fq.gz" --exclude "*.bmp" --exclude "*hyperspectral*"

# For training with hyperspectral data
aws s3 sync s3://yiming-qu/data/camii ~/data/camii --exclude "_*" --exclude "*.fastq.gz" --exclude "*.fq.gz" --exclude "*.bmp"
