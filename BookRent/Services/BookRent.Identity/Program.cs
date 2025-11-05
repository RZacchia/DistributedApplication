using BookRent.Identity.Api;
using BookRent.Identity.Infrastructure;
using BookRent.Identity.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

var cs = Environment.GetEnvironmentVariable("ConnectionStrings__IdentityDb")
         ?? builder.Configuration.GetConnectionString("IdentityDb");

builder.Services.AddDbContext<IdentityDbContext>(opt =>
    opt.UseSqlServer(cs, sql => sql.EnableRetryOnFailure(5)));
builder.Services.AddScoped<IIdentityRepository, IdentityRepository>();
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();



var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<IdentityDbContext>();
    db.Database.Migrate();
}




    app.MapScalarApiReference();
    app.MapOpenApi();

app.MapIdentityEndPoints();



app.Run();

