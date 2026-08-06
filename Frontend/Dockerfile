# Stage 1: Build the React application
FROM node:26-alpine AS build

WORKDIR /app

# Copy package files and install dependencies using npm
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the frontend code and build for production
COPY . ./
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Copy the optimized build output (Vite uses 'dist') to Nginx's html folder
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]