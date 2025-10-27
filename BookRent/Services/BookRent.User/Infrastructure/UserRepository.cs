using BookRent.User.Infrastructure.Interfaces;
using BookRent.User.Models;
using Microsoft.EntityFrameworkCore;

namespace BookRent.User.Infrastructure;

public class UserRepository : IUserRepository
{
    
    private UserDbContext Context { get; }

    public UserRepository(UserDbContext context)
    {
        Context = context;
    }
    
    public async Task<UserBaseData?> GetUserDataAsync(Guid userId)
    {
        return await Context.UserData.FirstOrDefaultAsync(x => x.UserId == userId);
    }

    public async Task<bool> UpdateUserDataAsync(UserBaseData userBaseData)
    {
        Context.UserData.Update(userBaseData);
        return await Context.SaveChangesAsync() == 1;
    }

    public async Task<bool> AddUserDataAsync(UserBaseData userBaseData)
    {
        await Context.UserData.AddAsync(userBaseData);
        return await Context.SaveChangesAsync() == 1;
    }

   

    public async Task<bool> AddUserFavouritesAsync(UserFavourites userFavourites)
    {
        Context.UserFavourites.Add(userFavourites);
        return await Context.SaveChangesAsync() == 1;
    }

    public async Task<bool> RemoveBookFromUserFavouritesAsync(UserFavourites userFavourites)
    {
        Context.UserFavourites.Remove(userFavourites);
        return await Context.SaveChangesAsync() == 1;
    }

    public async Task<bool> DeleteUserFavouritesListAsync(Guid userId)
    {
        Context.UserFavourites.RemoveRange(Context.UserFavourites
            .Where(x => x.UserId == userId));
        return await Context.SaveChangesAsync() > 0;
    }

    public async Task<List<UserFavourites>> GetUserFavouritesListAsync(Guid userId)
    {
        return await Context.UserFavourites.Where(u => u.UserId == userId)
            .ToListAsync();
    }
}