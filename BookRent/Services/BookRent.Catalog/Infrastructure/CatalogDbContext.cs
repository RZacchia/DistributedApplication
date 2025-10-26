using BookRent.Catalog.Model;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Catalog.Infrastructure;

public sealed class CatalogDbContext(DbContextOptions<CatalogDbContext> options) : DbContext(options)
{
    public DbSet<Book> Books => Set<Book>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        
    }
}