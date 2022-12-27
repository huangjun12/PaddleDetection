export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# SingleCard Debug
# python3.7 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml \
#                             --slim_config configs/slim/distill/ppyoloe_ld_distill.yml \
#                             --eval

# MultiCard
python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
                            tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml \
                            --slim_config configs/slim/distill/ppyoloe_ld_distill.yml \
                            --eval