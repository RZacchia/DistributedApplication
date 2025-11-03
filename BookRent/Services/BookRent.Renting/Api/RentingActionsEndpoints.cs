using BookRent.Renting.DTOs;
using BookRent.Renting.Infrastructure.Interfaces;
using BookRent.Renting.Models;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;

namespace BookRent.Renting.Api;

internal static class RentingActionsEndpoints
{
    internal static async Task<IResult> ReturnBooks([FromBody]  ReturnBookRequest request, IRentingRepository repo)
    {
        var success = await repo.ReturnBookAsync(request.BookId,  request.UserId);
        if (!success) return Results.BadRequest();
        return Results.Ok();
    }
    
    internal static async Task<Results<Ok<string>, BadRequest>> RentBooks([FromBody] RentBookRequest request, IRentingRepository repo)
    {
        var order = new RentedBook
        {
            OrderId = Guid.NewGuid(),
            UserId = request.UserId,
            BookId = request.BookId,
            RentedOn = DateTime.UtcNow,
            DueAt = DateTime.UtcNow.AddDays(14),

        };
        var orderId = await repo.RentBookAsync(order);
        return TypedResults.Ok(orderId.ToString());
    }
    
    internal static async Task<IResult> EditBookCounter([FromBody] EditBookCounterRequest request, IRentingRepository repo)
    {
        var success = await repo.EditBookCounterAsync(request.BookId, request.Counter);
        if(!success) return Results.BadRequest();
        return Results.Ok();
    }
}