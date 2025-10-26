using BookRent.Renting.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.Renting.Infrastructure;

public class RentingDbContext(DbContextOptions<RentingDbContext> options) : DbContext(options)
{
    public DbSet<BookCounter> BookCounters => Set<BookCounter>();
    public DbSet<RentedBook> RentedBooks => Set<RentedBook>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        
    }
    
}