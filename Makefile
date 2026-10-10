.PHONY: help build push demo kiosk down

IMAGE := jackcky/smktscnr-demo:latest
NAME  := smktscnr-demo
PORT  := 8501

help:
	@echo "Available targets:"
	@echo "  make help      # Show available targets"
	@echo "  make build     # Build the demo container image"
	@echo "  make push      # Push the demo container image to Docker Hub"
	@echo "  make demo      # Start the demo service in detached mode"
	@echo "  make kiosk     # Run the kiosk app locally (needs a webcam)"
	@echo "  make down      # Stop and remove the demo service"

build:
	docker build --platform linux/amd64 -f apps/demo/Dockerfile -t $(IMAGE) .

push:
	docker push $(IMAGE)

demo:
	docker run -d --platform linux/amd64 --name $(NAME) --restart unless-stopped --env-file config/.env -p $(PORT):8501 $(IMAGE)

kiosk:
	uv run python -m apps.kiosk.main

down:
	@docker rm -f $(NAME) >/dev/null 2>&1 || true
