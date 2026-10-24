# Frontend: build the React app, serve it with nginx, forward /api (HTTP + WebSocket) to the api service.
#   docker compose up web        -> http://localhost:8080
FROM node:20.18.1-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
