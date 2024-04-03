model="deit_small_patch16_224_ablation"
batchsize=256
img_size=224

merge_method='keep1'
merge_group_num=39
merge_group_size_1=1
merge_group_size_2=2
distance='euclidean'
token_pos_1=0
token_pos_2=1
score_gate=''

for merge_method in "keep1" "avg"
do
for distance in 'manhattan' 'cosine' 'euclidean'
do
# for merge_group_size_1 in {1,2}
# for token_pos_1 in {0,1}
# do
results_file=$model"_"$merge_method"_"$merge_group_num"_"$merge_group_size_1$merge_group_size_2"_"$distance"_"$token_pos_1$token_pos_2"_"$score_gate"_ben.csv"
python benchmark.py --model $model --bench inference -b $batchsize --num-warm-iter 100 --num-bench-iter 600 --img-size $img_size --results-file $results_file  --patch_merge --merge_method $merge_method --merge_group_num $merge_group_num --merge_group_size $merge_group_size_1 $merge_group_size_2 --distance $distance --token_pos $token_pos_1 $token_pos_2
# done
done
done