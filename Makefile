.PHONY: help build push demo kiosk down

COMPOSE_FILE := apps/compose.yaml
COMPOSE := docker compose -f $(COMPOSE_FILE)

help:
	@echo "Available targets:"
	@echo "  make help      # Show available targets"
	@echo "  make build     # Build the demo and kiosk container images"
	@echo "  make push      # Push the demo and kiosk container images to Docker Hub"
	@echo "  make demo      # Start the demo service in detached mode"
	@echo "  make kiosk     # Run the kiosk service using the existing image"
	@echo "  make down      # Stop and remove services defined in apps/compose.yaml"

build:
	$(COMPOSE) build

push:
	$(COMPOSE) push

demo:
	$(COMPOSE) up -d --no-build demo

kiosk:
	$(COMPOSE) run --rm kiosk

down:
	$(COMPOSE) down
