# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy only the csproj first (faster restores)
COPY Services/BookRent.User/BookRent.User.csproj Services/BookRent.User/
RUN dotnet restore Services/BookRent.User/BookRent.User.csproj

# Copy the rest of the project
COPY Services/BookRent.User/ Services/BookRent.User/

# Publish
WORKDIR /src/Services/BookRent.User
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8084
EXPOSE 8084
ENTRYPOINT ["dotnet", "BookRent.User.dll"]
