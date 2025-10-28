using BookRent.Catalog.DTO;
using BookRent.Catalog.Infrastructure.Interfaces;
using BookRent.Catalog.Model;
using Microsoft.AspNetCore.Mvc;

namespace BookRent.Catalog.Api;

internal static class CatalogEmployeeEndpoints
{
    internal static async Task<IResult> AddBook([FromBody] AddBookRequest request, IBookRepository repo)
    {
        try
        {
            var currentBook = new Book
            {
                Id = Guid.NewGuid(),
                Isbn = request.Isbn,
                Title = request.Name,
                Author = request.Author,
                Description = request.Description,
                IsVisible = true
            };
            bool success = await repo.AddBookAsync(currentBook);
            if (!success)
            {
                return Results.BadRequest();
            }
        }
        catch (Exception ex)
        {
            return Results.InternalServerError(ex);
        }
        return Results.Ok();
    }
    
    internal static async Task<IResult> RemoveBook(Guid id,  IBookRepository repo)
    {
        var success = await repo.DeleteBookAsync(id);
        if (!success) return Results.BadRequest();
        return Results.Ok();
    }
    
    internal static async Task<IResult> UpdateBook([FromBody] UpdateBookRequest request,  IBookRepository repo)
    {
        var result = await repo.GetBookAsync(request.Id);
        if (result == null) return Results.BadRequest();
        var success = await repo.UpdateBookAsync(result);
        if (!success) return Results.BadRequest();
        return Results.Ok();
    }
}