using BookRent.Renting.Api;
using BookRent.Renting.Infrastructure;
using BookRent.Renting.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

var cs = Environment.GetEnvironmentVariable("ConnectionStrings__RentingDb")
         ?? builder.Configuration.GetConnectionString("RentingDb");
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddDbContext<RentingDbContext>(opt => 
{
    opt.UseSqlServer(cs, sql => sql.EnableRetryOnFailure(5));

});
builder.Services.AddScoped<IRentingRepository, RentingRepository>();


var app = builder.Build();


using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<RentingDbContext>();
    db.Database.Migrate();
}


    app.MapScalarApiReference();
    app.MapOpenApi();

app.MapRentingEndpoints();



app.Run();