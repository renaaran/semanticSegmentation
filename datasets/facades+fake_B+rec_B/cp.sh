#set -x
cnt=605
type='train'
for ((i=303; i < 605; i++))
do
	echo ${i} ${cnt}
	cp ../facades+rec_B/${type}/${i}.jpg ./${type}/${cnt}.jpg
	cp ../facades+rec_B/${type}/${i}.png ./${type}/${cnt}.png
	cnt=$((cnt+1))
done
echo $(ls -1 ${type}/*.jpg|wc -l) $(ls -1 ${type}/*.jpg|cut -d/ -f2|sort -un|tail -1|cut -d. -f1)
echo $(ls -1 ${type}/*.png|wc -l) $(ls -1 ${type}/*.png|cut -d/ -f2|sort -un|tail -1|cut -d. -f1)
