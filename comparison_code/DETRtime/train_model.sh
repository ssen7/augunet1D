date=$(date +%y-%m-%d-%H-%M)    
modelname=$'mice_eeg_final_v3'

python main.py \
	--data_path /standard/ivy-hip-rderi/ss4yd/DETRtime \
	--backbone inception_time \
	--lr_backbone 1e-4 \
	--nb_filters 16 \
	--use_residual True \
	--num_classes 2 \
	--backbone_depth 6 \
	--batch_size 32 \
	--bbox_loss_coef 10 \
	--giou_loss_coef 2 \
	--eos_coef 0.4 \
	--hidden_dim 128 \
	--dim_feedforward 512 \
	--dropout 0.1 \
	--wandb_dir movie \
	--num_queries 2 \
	--lr_drop 50 \
	--num_workers 10 \
	--epochs 10 \
	--timestamps 2000 \
	--timestamps_output 2000 \
	--output_dir ./runs/"$modelname" &
