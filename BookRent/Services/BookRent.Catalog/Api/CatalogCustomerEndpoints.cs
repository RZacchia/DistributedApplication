using BookRent.Catalog.DTO;
using BookRent.Catalog.Infrastructure.Interfaces;
using BookRent.Catalog.Model;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;

namespace BookRent.Catalog.Api;

internal static class CatalogCustomerEndpoints
{
    internal static async Task<Results<Ok<List<Book>>, NoContent>> GetAllBooks(HttpRequest request,
        IBookRepository repo)
    {
        var result = await repo.GetBooksAsync();
        if (!result.Any())
        {
            return TypedResults.NoContent();
        }
        return TypedResults.Ok(result);
    }
    
    internal static async Task<Results<Ok<Book>, NoContent>> GetBook(Guid id, HttpRequest request,  IBookRepository repo)
    {
        var result = await repo.GetBookAsync(id);
        if (result is null)
        {
            return TypedResults.NoContent();
        }
        return TypedResults.Ok(result);
    }
    
    internal static async Task<Results<Ok<List<Book>>, NoContent>> SearchBook([FromBody] BookSearchRequest request, IBookRepository repo)
    {
        var result = await repo.GetBooksByNameAsync(request.Name);
        if (result.Count == 0)
        {
            return TypedResults.NoContent();
        }
        return TypedResults.Ok(result);
    }
}