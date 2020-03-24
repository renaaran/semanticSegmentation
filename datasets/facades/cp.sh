cnt=1
type='train'
for i in $(cat ../small/index/${type}.txt):
do
	echo ${cnt}
	n=$(echo $i|cut -d_ -f2|cut -d- -f1|cut -db -f2|sed -e 's/^[0]*//')
	cp /home/CAMPUS/180178991/Pictures/CMP_facade_DB_base/base/${n}.jpg ./${type}/${cnt}.jpg
	cp /home/CAMPUS/180178991/Pictures/CMP_facade_DB_base/base/${n}.png ./${type}/${cnt}.png
	cnt=$((cnt+1))
done
