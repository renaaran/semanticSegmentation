#set -x
cnt=$(ls -1 train/|sort -un|tail -1|cut -d. -f1)
cnt=$((cnt+1))
type='train'
for i in $(ls -1 ./test_latest/images/*fake_B*)
do
	echo ${cnt}
	n=$(echo $i|xargs basename|cut -d_ -f1)
	convert $i ./${type}/${cnt}.jpg
	cp ./train/${n}.png ./${type}/${cnt}.png
	cnt=$((cnt+1))
done
echo $(ls -1 ${type}/*.jpg|wc -l) $(ls -1 ${type}/*.jpg|cut -d/ -f2|sort -un|tail -1|cut -d. -f1)
echo $(ls -1 ${type}/*.png|wc -l) $(ls -1 ${type}/*.png|cut -d/ -f2|sort -un|tail -1|cut -d. -f1)