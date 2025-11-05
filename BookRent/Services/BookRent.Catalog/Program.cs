using BookRent.Catalog.Api;
using BookRent.Catalog.Infrastructure;
using BookRent.Catalog.Infrastructure.Interfaces;
using Microsoft.EntityFrameworkCore;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);




var cs = Environment.GetEnvironmentVariable("ConnectionStrings__CatalogDb")
         ?? builder.Configuration.GetConnectionString("CatalogDb");

builder.Services.AddDbContext<CatalogDbContext>(opt =>
    opt.UseSqlServer(cs, sql => sql.EnableRetryOnFailure(5)));
builder.Services.AddScoped<IBookRepository, BookRepository>();
builder.Services.AddOpenApi();


var app = builder.Build();


using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<CatalogDbContext>();
    db.Database.Migrate();
}



    app.MapOpenApi();
    app.MapScalarApiReference();


app.MapCatalogEndpoints();


app.Run();
