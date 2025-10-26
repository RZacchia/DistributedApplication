using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace BookRent.Catalog.Infrastructure;

public sealed class CatalogDbContextFactory : IDesignTimeDbContextFactory<CatalogDbContext>
{
    public CatalogDbContext CreateDbContext(string[] args)
    {
        // Dev connection (will switch to Docker later)
        var cs = Environment.GetEnvironmentVariable("CATALOG_CS")
                 ?? "Server=localhost,1433;Database=CatalogDb;User Id=sa;Password=Your_password123;TrustServerCertificate=True;";

        var opts = new DbContextOptionsBuilder<CatalogDbContext>()
            .UseSqlServer(cs)
            .Options;

        return new CatalogDbContext(opts);
    }
}