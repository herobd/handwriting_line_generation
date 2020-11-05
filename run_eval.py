#!/bin/bash


#SBATCH --time=0:30:00   # walltime
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -J "eval"
#SBATCH --mem-per-cpu=2048M
#SBATCH --mail-user=herobd@gmail.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#xxxxSBATCH -C 'pascal'
#xxxSBATCH --qos=standby   
#SBATCH --requeue

#130:00:00

export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module load cuda/10.1
module load cudnn/7.6
cd ~/hw_with_style
source deactivate
source activate c10

python new_eval.py -c saved/fontNAF32_ocr_softmax_1huge/checkpoint-iteration500000.pth -g 0 -f cf_test_on_FUNSD.json
#python new_eval.py -c saved/allfontSmall_ocr_softmax_1huge/checkpoint-iteration650000.pth -g 0 -f cf_test_on_FUNSD.json

#saved/IAMslant_noMask_bigStyleEx_GANMedMT_autoAEMoPrcp2tightNoPyr_balB_hCF0.75_sMG

#python get_styles.py -c saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-iteration175000.pth -g 0 -d direiogtjaerjkl -T

#python interpolate.py -c saved/IAM_gen_short/checkpoint-iteration2000.pth -g 0 -d gold_mturk -a style_loc=saved/IAM_gen_short/test_styles_2000.pkl -r choice=t,num_inst=8,start_index=70 -T
#python interpolate.py -c saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-iteration175000.pth -g 0 -d final2_mturk -a style_loc=saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG/test_styles_175000.pkl -r choice=t,num_inst=35 -T


#rm -r out/*
###


#python new_eval.py -c saved/IAMslant_noMaskNoSpace_bigStyleEx_GANMedMT_noAuto_balB_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [gen] -d out -a data_loader=short=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMaskNoSpace_bigStyleExAppend_GANMedMT_noAuto_balB_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [gen] -d out -a data_loader=short=1 -g 0 -v 1

#python new_eval.py -c saved/IAMslant_noMask_DNoMid_charSpecSingleAppend_GANMedMT_autoPrcpNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-iteration175000.pth -n 12 -e [recon_pred_space,gen] -d out -a data_loader=short=1,data_loader=batch_size=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMask_aBatch1_charSpecSingleAppend_GANMedMT_autoPrcpNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-iteration175000.pth -n 12 -e [recon_pred_space,gen] -d out -a data_loader=short=1,data_loader=batch_size=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoNoPrcpUseGen_balB_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [recon_pred_space,gen] -d out -a data_loader=short=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoNoPixPrcpNewCTCUseGen_balB_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [recon_pred_space,gen] -d out -a data_loader=short=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcpNewCTCUseGen_balOrig_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [recon_pred_space,gen] -d out -a data_loader=short=1 -g 0 -v 1
#python new_eval.py -c saved/IAMslant_noMask_bigStyleExAppend_noGAN_autoAEMoPrcp2tightNewCTC_balB_hCF0.75_sMG/checkpoint-latest.pth -n 8 -e [recon_pred_space,gen] -d out -a data_loader=short=1 -g 0 -v 1
