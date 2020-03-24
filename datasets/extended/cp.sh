cnt=1
type='test'
path='/home/CAMPUS/180178991/Pictures/CMP_facade_DB_extended/extended'
cnt=$(ls -1 ${path}/[0-9]*.jpg|wc -l)
for ((i=1; i < $cnt; ++i))
do
	echo ${i}
	cp ${path}/${i}.jpg ./${type}/${i}.jpg
	cp ${path}/${i}.png ./${type}/${i}.png
done
