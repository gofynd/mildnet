source settings.cfg
source $1
echo $1
gcloud ml-engine local train \
--module-name trainer.task \
--package-path trainer/ \
--job-dir $MILDNET_JOB_DIR$model_name \
-- \
--data-path=$MILDNET_DATA_PATH \
--model-id=$model_id \
--loss=$loss \
--optimizer=$optimizer \
--train-csv=$train_csv \
--val-csv=$val_csv \
--train-epocs=$train_epocs \
--lr=$lr