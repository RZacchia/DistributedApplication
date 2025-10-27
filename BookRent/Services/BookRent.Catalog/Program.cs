using BookRent.Catalog.Api;
using BookRent.Catalog.Infrastructure;
using BookRent.Catalog.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);


// Connection string (for dev; will override with Docker env later)


builder.Services.AddDbContext<CatalogDbContext>();
builder.Services.AddScoped<IBookRepository, BookRepository>();
// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();
app.MapCatalogEndpoints();


app.Run();
