# Stage 1: build
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
# Install deps (pick your manager; here npm)
RUN npm install

COPY . .
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:1.27-alpine

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Basic nginx config (SPA-friendly)
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
