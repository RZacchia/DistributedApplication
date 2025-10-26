using BookRent.User.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.User.Infrastructure;

public class UserDbContext(DbContextOptions<UserDbContext> options) : DbContext(options)
{
    public DbSet<UserBaseData> BookCounters => Set<UserBaseData>();
    public DbSet<UserFavourites> RentedBooks => Set<UserFavourites>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        
    }
    
}