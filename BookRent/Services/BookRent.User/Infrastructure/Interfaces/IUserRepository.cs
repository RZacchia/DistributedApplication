using BookRent.User.Models;

namespace BookRent.User.Infrastructure.Interfaces;

public interface IUserRepository
{
    Task<UserBaseData?> GetUserDataAsync(Guid userId);
    Task<bool> UpdateUserDataAsync(UserBaseData userBaseData);
    Task<bool> AddUserDataAsync(UserBaseData userBaseData);
    
    Task<bool> AddUserFavouritesAsync(UserFavourites userFavourites);
    Task<bool> RemoveBookFromUserFavouritesAsync(UserFavourites userFavourites);
    Task<bool> DeleteUserFavouritesListAsync(Guid userId);
    Task<List<UserFavourites>> GetUserFavouritesListAsync(Guid userId);
}