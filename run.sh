CUDA_VISIBLE_DEVICES="1,2,3,4" python -m torch.distributed.run --nproc_per_node 4 --master_port 29501 DDP_main.py \
  --lr 5e-5 \
  --batch_size 128 \
  --from_scratch false \
  --epochs 10 \
  --eval_steps 240 \
  --logging_steps 100