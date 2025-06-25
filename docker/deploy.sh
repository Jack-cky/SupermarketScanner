# build image
docker build --no-cache --platform linux/amd64 -t jackcky/supermarketscanner:v0 -f docker/Dockerfile .

# tag image
docker tag jackcky/supermarketscanner:v0 jackcky/supermarketscanner:latest

# push to Docker Hub
docker push jackcky/supermarketscanner:latest

# run container
docker run --name supermarketscanner --platform linux/amd64 -p 8501:8501 jackcky/supermarketscanner:latest
