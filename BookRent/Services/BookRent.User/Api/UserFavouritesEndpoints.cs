using BookRent.User.DTO;
using BookRent.User.Infrastructure.Interfaces;
using BookRent.User.Models;
using Microsoft.AspNetCore.Http.HttpResults;

namespace BookRent.User.Api;

internal static class UserFavouritesEndpoints
{
    internal static async Task<Results<Ok<UserFavouriteResponse>,NoContent, BadRequest>> GetFavourites(Guid id, IUserRepository repo)
    {
        var userFavourites = await repo.GetUserFavouritesListAsync(id);
        if (userFavourites.Count == 0) return TypedResults.NoContent();

        var response = new UserFavouriteResponse(userFavourites.Select(x => x.BookId).ToList());
        
        return TypedResults.Ok(response);
    }
    
    internal static IResult AddFavourite(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    internal static async Task<IResult> RemoveFavourite(Guid userId, Guid bookId, IUserRepository repo)
    {
        var queryParam = new UserFavourites
        {
            UserId = userId,
            BookId = bookId
        };
        var success = await repo.RemoveBookFromUserFavouritesAsync(queryParam);
        if (!success) return Results.BadRequest();
        
        return Results.Ok();
    }
}