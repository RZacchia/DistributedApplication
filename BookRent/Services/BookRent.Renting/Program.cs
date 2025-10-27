using BookRent.Renting.Api;
using BookRent.Renting.Infrastructure;
using BookRent.Renting.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddDbContext<RentingDbContext>();
builder.Services.AddScoped<IRentingRepository, RentingRepository>();


var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<RentingDbContext>();
    db.Database.Migrate();
}


app.MapRentingEndpoints();
if (app.Environment.IsDevelopment())
{
    app.MapScalarApiReference();
    app.MapOpenApi();
}

app.UseHttpsRedirection();



app.Run();