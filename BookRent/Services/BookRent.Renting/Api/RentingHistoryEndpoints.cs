using BookRent.Renting.Infrastructure.Interfaces;
using BookRent.Renting.Models;
using Microsoft.AspNetCore.Http.HttpResults;

namespace BookRent.Renting.Api;

internal static class RentingHistoryEndpoints
{
    internal static async Task<Results<Ok<List<RentedBook>>, NoContent>> GetRentedBooks(Guid userId, IRentingRepository repo)
    {
        var result = await repo.GetRentHistoryAsync(userId);
        if (result.Count == 0) return TypedResults.NoContent();
        return TypedResults.Ok(result);
    }
    
    internal static async Task<Results<Ok<List<RentedBook>>, NoContent>> GetCurrentlyRentBooks(IRentingRepository repo)
    {
        var result = await repo.GetAllRentsAsync();
        if (!result.Any())
        {
            return TypedResults.NoContent();
        }
        return TypedResults.Ok(result);
    }
    

}