# =========================
# Build stage
# =========================
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy only the csproj first (faster restores)
COPY Services/BookRent.Renting/BookRent.Renting.csproj Services/BookRent.Renting/
RUN dotnet restore Services/BookRent.Renting/BookRent.Renting.csproj

# Copy the rest of the project
COPY Services/BookRent.Renting/ Services/BookRent.Renting/

# Publish
WORKDIR /src/Services/BookRent.Renting
RUN dotnet publish -c Release -o /app/publish

# =========================
# Runtime stage
# =========================
FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8083
EXPOSE 8083
ENTRYPOINT ["dotnet", "BookRent.Renting.dll"]
